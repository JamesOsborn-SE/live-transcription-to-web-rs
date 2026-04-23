FROM rust:trixie AS builder

# Install dependencies required by cpal (ALSA) and OpenSSL/Network stuff
RUN apt-get update && apt-get install -y \
    pkg-config \
    libasound2-dev \
    build-essential

WORKDIR /usr/src/app
COPY . .

# Build the release binary
RUN cargo build --release

# --- Stage 2: Runtime ---
FROM debian:trixie

# Install ALSA runtime libraries
RUN apt-get update && apt-get install -y \
    libasound2 \
    alsa-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the compiled binary from the builder stage
COPY --from=builder /usr/src/app/target/release /app/live-transcription

# Copy your web assets if they aren't embedded into the binary
COPY ./index.html /app/index.html 

# Expose the Axum web server port
EXPOSE 3000

# Run the app
CMD ["./live-transcription"]