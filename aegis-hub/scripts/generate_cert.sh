#!/usr/bin/env bash
# generate_cert.sh — Generates self-signed TLS cert/key pair for dev and bench testing

set -euo pipefail

CERT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/certs"
mkdir -p "$CERT_DIR"

CERT_FILE="$CERT_DIR/hub.crt"
KEY_FILE="$CERT_DIR/hub.key"

if [ -f "$CERT_FILE" ] && [ -f "$KEY_FILE" ]; then
    echo "Certificates already exist at $CERT_DIR"
    exit 0
fi

echo "Generating self-signed TLS certificate for AEGIS Hub..."
openssl req -x509 -newkey rsa:4096 -nodes \
    -keyout "$KEY_FILE" \
    -out "$CERT_FILE" \
    -days 365 \
    -subj "/CN=aegis-hub/O=Project AEGIS/C=US"

chmod 600 "$KEY_FILE"
chmod 644 "$CERT_FILE"

echo "Certificate generated successfully:"
echo "  Cert: $CERT_FILE"
echo "  Key:  $KEY_FILE"
