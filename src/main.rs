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
use ort::ep::OpenVINO;
use parakeet_rs::{ExecutionConfig, ExecutionProvider, Nemotron};
use rtrb::{Consumer, Producer, RingBuffer};
use std::{convert::Infallible, time::Duration};
use tokio::sync::broadcast;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::BroadcastStream;
use tower_http::cors::CorsLayer;

// --- Configuration Constants ---
const SAMPLE_RATE: u32 = 16000;
const CHANNELS: u16 = 1;
const RING_BUFFER_SIZE: usize = SAMPLE_RATE as usize * 10;
const CHUNK_SIZE: usize = 8960; // 560ms at 16kHz (Standard chunk size for Nemotron streaming)

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
    tracing::info!("Starting Live Transcriber (Parakeet-RS)...");

    // 1. Initialize Broadcast Channel
    let (tx, _rx) = broadcast::channel::<String>(16);
    let app_state = AppState { tx: tx.clone() };

    // 2. Initialize Lock-Free Ring Buffer
    let (audio_prod, audio_cons) = RingBuffer::<f32>::new(RING_BUFFER_SIZE);

    // 3. Start Inference Thread (Dedicated OS thread for ML workload)
    let _inference_handle = std::thread::spawn(move || {
        run_inference_loop(audio_cons, tx).expect("Inference thread crashed");
    });

    // 4. Start Audio Capture Thread (Isolate CPAL from Tokio!)
    // We use a channel just to block until the stream is ready, so we don't proceed too fast
    let (stream_tx, stream_rx) = std::sync::mpsc::channel();

    let _audio_handle = std::thread::spawn(move || {
        // Start the stream inside this dedicated OS thread
        match start_audio_stream(audio_prod) {
            Ok(stream) => {
                if let Err(e) = stream.play() {
                    tracing::error!("Failed to play stream: {}", e);
                } else {
                    tracing::info!("Microphone stream started on dedicated thread.");
                    // Tell main thread we succeeded
                    let _ = stream_tx.send(Ok(stream));

                    // Keep the thread alive so the stream doesn't drop
                    loop {
                        std::thread::park();
                    }
                }
            }
            Err(e) => {
                let _ = stream_tx.send(Err(e));
            }
        }
    });

    // Wait for the audio thread to successfully initialize
    let _audio_stream_keepalive = stream_rx.recv().expect("Audio thread died")?;

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

fn calculate_rms(samples: &[f32]) -> f32 {
    let sq_sum: f32 = samples.iter().map(|&s| s * s).sum();
    if samples.is_empty() {
        return 0.0;
    }
    (sq_sum / samples.len() as f32).sqrt()
}

/// Dedicated blocking loop for Parakeet-RS processing
fn run_inference_loop(
    mut consumer: Consumer<f32>,
    tx: broadcast::Sender<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    tracing::info!("Initializing Parakeet-RS Nemotron Engine...");

    let config = ExecutionConfig::new()
        .with_custom_configure(|builder| {
            let ov_ep = OpenVINO::default()
                .with_device_type("HETERO:GPU,CPU")
                .with_dynamic_shapes(false)
                .build();

            let configured_builder = builder.with_execution_providers([ov_ep])?;

            // 2. Return the successfully configured builder wrapped in Ok()
            Ok(configured_builder)
        })
        .with_execution_provider(ExecutionProvider::Cpu);

    let mut model = Nemotron::from_pretrained("./nemotron", Some(config))?;

    let mut audio_buffer = Vec::with_capacity(CHUNK_SIZE * 2);
    let mut completed_text = String::new();

    // --- Silence Detection State ---
    let mut silent_chunks = 0;
    let mut needs_newline = false;
    const SILENCE_THRESHOLD: f32 = 0.05; // Tune this: lower = more sensitive to background noise
    const CHUNKS_FOR_NEWLINE: usize = 1; // ~1.1 seconds of silence (2 * 560ms)

    loop {
        // Drain available samples into the buffer
        while let Ok(sample) = consumer.pop() {
            audio_buffer.push(sample);
        }

        // Process exactly CHUNK_SIZE samples at a time
        if audio_buffer.len() >= CHUNK_SIZE {
            let chunk: Vec<f32> = audio_buffer.drain(..CHUNK_SIZE).collect();

            // Check audio volume to detect pauses
            let rms = calculate_rms(&chunk);
            if rms < SILENCE_THRESHOLD {
                silent_chunks += 1;
                if silent_chunks >= CHUNKS_FOR_NEWLINE {
                    needs_newline = true;
                }
            } else {
                silent_chunks = 0; // Reset if speech is detected
            }

            // Perform cache-aware transcription on the chunk
            if let Ok(text) = model.transcribe_chunk(&chunk) {
                if !text.is_empty() {
                    if needs_newline {
                        // Check if the new chunk actually contains words, or just leftover punctuation/spaces
                        let has_alphanumeric = text.chars().any(|c| c.is_alphanumeric());

                        if !has_alphanumeric {
                            // It's JUST leftover punctuation (e.g. "." or " ?").
                            // Attach it to the previous line and KEEP the newline pending for the next actual word.
                            completed_text.push_str(&text);
                        } else {
                            // The chunk contains words, but might start with punctuation (e.g. ". How are you")
                            // Let's split off any leading punctuation/spaces
                            let first_alpha_idx = text.find(|c: char| c.is_alphanumeric()).unwrap();
                            let (leading_punc, rest) = text.split_at(first_alpha_idx);

                            // Attach the lingering punctuation to the previous line
                            completed_text.push_str(leading_punc);

                            // Insert the paragraph break (but avoid doing it at the very start of the doc)
                            if !completed_text.trim().is_empty()
                                && !completed_text.ends_with("\n\n")
                            {
                                completed_text.push_str("\n");
                            }

                            // Append the actual words
                            completed_text.push_str(rest);
                            needs_newline = false; // We successfully inserted the newline
                        }
                    } else {
                        completed_text.push_str(&text);
                    }

                    let payload = serde_json::json!({
                        "completed": completed_text,
                    })
                    .to_string();

                    if tx.receiver_count() > 0 {
                        let _ = tx.send(payload);
                    }
                }
            }
        } else {
            // Sleep briefly to avoid busy-waiting
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
