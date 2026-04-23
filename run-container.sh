docker run -it --rm \
  -p 3000:3000 \
  --device /dev/dri \
  --device /dev/snd \
  --group-add audio \
  -v $(pwd)/nemotron:/app/nemotron \
  live-transcriber