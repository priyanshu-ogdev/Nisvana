#!/usr/bin/env bash
# install_system_deps.sh — Install ALSA, PortAudio, and system libraries
# Run as: sudo bash scripts/install_system_deps.sh

set -euo pipefail

echo "=== Installing ALSA / PortAudio system dependencies ==="
apt-get update -q
apt-get install -y \
    alsa-utils \
    alsa-tools \
    libasound2-dev \
    portaudio19-dev \
    libsndfile1-dev \
    libopus-dev \
    ffmpeg \
    python3-pyaudio

# Enable I2S audio (for Pi HATs, optional)
# echo "dtparam=audio=on" >> /boot/firmware/config.txt

echo "=== Done. Reboot if needed. ==="
