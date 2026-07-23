#!/bin/bash
# DS4 Server launcher — starts in terminal for live monitoring
# Edit this script to change parameters

cd /home/tosol/ds4

numactl --interleave=all ./ds4-server \
  --rocm \
  -c 256000 \
  -m DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --ssd-streaming \
  --ssd-streaming-cache-experts 32GB
