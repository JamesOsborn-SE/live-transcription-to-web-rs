# Live transcription to web

## Purpose

Generate low latency, live transcriptions, that may be viewed in a browser on the local network for folks.

## Install system deps

### Ubuntu 24.04

#### OpenVino

```shell
sudo apt install -y intel-opencl-icd intel-level-zero-gpu level-zero 
```

#### Sound

```shell
sudo apt install -y build-essential pkg-config libasound2-dev libssl-dev
```

## Get model files

```shell
mkdir model
cd model
wget https://huggingface.co/csukuangfj/sherpa-onnx-nemotron-speech-streaming-en-0.6b-int8-2026-01-14/resolve/main/encoder.int8.onnx
wget https://huggingface.co/csukuangfj/sherpa-onnx-nemotron-speech-streaming-en-0.6b-int8-2026-01-14/resolve/main/decoder.int8.onnx
wget https://huggingface.co/csukuangfj/sherpa-onnx-nemotron-speech-streaming-en-0.6b-int8-2026-01-14/resolve/main/joiner.int8.onnx
wget https://huggingface.co/csukuangfj/sherpa-onnx-nemotron-speech-streaming-en-0.6b-int8-2026-01-14/resolve/main/tokens.txt
```

## run

```shell
cargo run
```

replace with ip of device find ip: `ip addr`
[Live transcription](http://192.168.1.252:3000/)

## Todos

### Code

- read model config from file
- retry if no mic found
- exit if model isn't loaded
- add dockers for different Providers
- look into more accurate models

### Meatbags

- read and understand the code
- find co-maintainer

## About

Made with Gemini Pro 3.1
