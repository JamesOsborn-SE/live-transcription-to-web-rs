use axum::{
    Router,
    extract::State,
    response::{
        Html,
        sse::{Event, Sse},
    },
    routing::get,
};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use futures::stream::Stream;
use rtrb::{Consumer, Producer, RingBuffer};
use sherpa_onnx::{OnlineRecognizer, OnlineRecognizerConfig};
use std::{convert::Infallible, time::Duration};
use tokio::sync::broadcast;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::BroadcastStream;
use tower_http::cors::CorsLayer;

// --- Configuration Constants ---
const SAMPLE_RATE: u32 = 16000;
const CHANNELS: u16 = 1;
const RING_BUFFER_SIZE: usize = SAMPLE_RATE as usize * 10;

// Shared state for Axum Web Server
#[derive(Clone)]
struct AppState {
    tx: broadcast::Sender<String>,
}

async fn index_handler() -> Html<&'static str> {
    Html(include_str!("../index.html"))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    tracing::info!("Starting Live Transcriber (Sherpa-ONNX)...");

    // 1. Initialize Broadcast Channel
    let (tx, _rx) = broadcast::channel::<String>(16);
    let app_state = AppState { tx: tx.clone() };

    // 2. Initialize Lock-Free Ring Buffer
    let (audio_prod, audio_cons) = RingBuffer::<f32>::new(RING_BUFFER_SIZE);

    // 3. Start Inference Thread (Dedicated OS thread for ML workload)
    let _inference_handle = std::thread::spawn(move || {
        run_inference_loop(audio_cons, tx).expect("Inference thread crashed");
    });

    // 4. Start Audio Capture Thread (cpal)
    let audio_stream = start_audio_stream(audio_prod)?;
    audio_stream.play()?;
    tracing::info!("Microphone stream started.");

    // 5. Start Network Thread (Tokio + Axum)
    let app = Router::new()
        .route("/", get(index_handler))
        .route("/events", get(sse_handler))
        .with_state(app_state)
        .layer(CorsLayer::permissive());

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    tracing::info!("SSE server listening on http://0.0.0.0:3000/events");

    // 6. Graceful Shutdown Handling
    tokio::select! {
        _ = axum::serve(listener, app) => {},
        _ = tokio::signal::ctrl_c() => {
            tracing::info!("Ctrl+C received, shutting down gracefully...");
        }
    }

    audio_stream.pause()?;
    drop(audio_stream);
    tracing::info!("Audio stream closed. Exiting.");

    Ok(())
}

/// Sets up the CPAL audio capture
fn start_audio_stream(
    mut producer: Producer<f32>,
) -> Result<cpal::Stream, Box<dyn std::error::Error>> {
    let host = cpal::default_host();
    let device = host.default_input_device().ok_or("No input device found")?;
    tracing::info!("Using input device: {}", device.description()?);

    let config = cpal::StreamConfig {
        channels: CHANNELS,
        sample_rate: SAMPLE_RATE,
        buffer_size: cpal::BufferSize::Default,
    };

    let stream = device.build_input_stream(
        &config,
        move |data: &[f32], _: &_| {
            // Push audio into the wait-free ring buffer
            for &sample in data {
                let _ = producer.push(sample);
            }
        },
        |err| tracing::error!("Audio stream error: {}", err),
        None,
    )?;

    Ok(stream)
}

/// Dedicated blocking loop for Sherpa-ONNX processing
fn run_inference_loop(
    mut consumer: Consumer<f32>,
    tx: broadcast::Sender<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut config = OnlineRecognizerConfig::default();
    config.model_config.provider = Some("openvino".to_string());

    config.model_config.model_type = Some("transducer".to_string());
    config.model_config.transducer.encoder = Some("model/encoder.int8.onnx".to_string());
    config.model_config.transducer.decoder = Some("model/decoder.int8.onnx".to_string());
    config.model_config.transducer.joiner = Some("model/joiner.int8.onnx".to_string());
    // config.model_config.transducer.encoder = Some("model/encoder.onnx".to_string());
    // config.model_config.transducer.decoder = Some("model/decoder.onnx".to_string());
    // config.model_config.transducer.joiner = Some("model/joiner.onnx".to_string());
    config.model_config.tokens = Some("model/tokens.txt".to_string());
    config.model_config.nemo_ctc.model = None;

    config.enable_endpoint = true;

    tracing::info!("Initializing Sherpa-ONNX Engine...");
    let recognizer =
        OnlineRecognizer::create(&config).ok_or("Failed to create OnlineRecognizer")?;
    let mut stream = recognizer.create_stream();

    let chunk_size = (SAMPLE_RATE / 10) as usize;
    let mut audio_buffer = Vec::with_capacity(chunk_size);

    let mut last_active = String::new();
    let mut completed_text = String::new();

    loop {
        while let Ok(sample) = consumer.pop() {
            audio_buffer.push(sample);
        }

        if audio_buffer.len() >= chunk_size {
            stream.accept_waveform(SAMPLE_RATE as i32, &audio_buffer);
            audio_buffer.clear();

            while recognizer.is_ready(&mut stream) {
                recognizer.decode(&mut stream);
            }

            let result = recognizer.get_result(&mut stream).unwrap();
            let is_endpoint = recognizer.is_endpoint(&mut stream);

            let mut active_text = result.text.clone();
            let mut state_changed = false;

            // If the model detects a pause in speech, finalize the sentence
            if is_endpoint {
                if !active_text.is_empty() {
                    completed_text.push_str(&active_text);
                    completed_text.push_str("\n");
                    state_changed = true;
                }
                // Clear the active text since the sentence is finished
                active_text.clear();
                recognizer.reset(&mut stream);
            }

            // Flag if the live transcription changed
            if active_text != last_active {
                state_changed = true;
            }

            // If either active or completed text changed, broadcast the new state
            if state_changed {
                // Package the state as a JSON string
                let payload = serde_json::json!({
                    "completed": completed_text,
                    "active": active_text
                })
                .to_string();

                if tx.receiver_count() > 0 {
                    let _ = tx.send(payload);
                }
                last_active = active_text;
            }
        } else {
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

/// Axum handler to establish SSE connections
async fn sse_handler(
    State(state): State<AppState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let rx = state.tx.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(|msg| match msg {
        Ok(text) => Some(Ok(Event::default().data(text))),
        Err(_) => None,
    });

    Sse::new(stream).keep_alive(axum::response::sse::KeepAlive::new())
}
