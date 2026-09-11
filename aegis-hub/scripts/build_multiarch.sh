#!/usr/bin/env bash
# scripts/build_multiarch.sh — Build and push multi-arch AEGIS images
#
# Builds both aegis-node (arm64 + amd64) and aegis-hub (amd64).
# Uses docker buildx. Requires Docker Desktop or buildx with QEMU.
#
# Usage:
#   REGISTRY=ghcr.io/your-team bash scripts/build_multiarch.sh
#   REGISTRY=ghcr.io/your-team TAG=v1.2.3 bash scripts/build_multiarch.sh
#
# Topology rule:
#   aegis-node  → linux/arm64 (Pi), linux/amd64 (CI only — no real audio)
#   aegis-hub   → linux/amd64 (any x86 server or Pi amd64)
#
# Prefer native Pi builds for arm64 (zero emulation risk):
#   On Pi: docker build -f Dockerfile.node -t aegis-node:arm64-latest .

set -euo pipefail

REGISTRY="${REGISTRY:-ghcr.io/team-aegis}"
TAG="${TAG:-latest}"
PUSH="${PUSH:-true}"

NODE_IMAGE="${REGISTRY}/aegis-node:${TAG}"
HUB_IMAGE="${REGISTRY}/aegis-hub:${TAG}"

echo "=== AEGIS Multi-Arch Build ==="
echo "  Node image : ${NODE_IMAGE}"
echo "  Hub image  : ${HUB_IMAGE}"
echo "  Push       : ${PUSH}"
echo ""

# Ensure buildx builder with multi-arch support exists
if ! docker buildx inspect aegis-builder &>/dev/null; then
    echo "Creating buildx builder 'aegis-builder'..."
    docker buildx create --name aegis-builder --use --platform linux/arm64,linux/amd64
else
    docker buildx use aegis-builder
fi

# aegis-node (built from aegis-backend/)
echo "Building aegis-node (arm64 + amd64)..."
BACKEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/aegis-backend"
if [ ! -d "${BACKEND_DIR}" ]; then
    echo "ERROR: aegis-backend not found at ${BACKEND_DIR}"
    exit 1
fi

BUILD_CMD="docker buildx build \
    --platform linux/arm64,linux/amd64 \
    -f ${BACKEND_DIR}/Dockerfile.node \
    -t ${NODE_IMAGE}"

if [ "${PUSH}" = "true" ]; then
    BUILD_CMD="${BUILD_CMD} --push"
else
    BUILD_CMD="${BUILD_CMD} --load"
fi

${BUILD_CMD} "${BACKEND_DIR}"

# aegis-hub (built from aegis-hub/)
echo "Building aegis-hub (amd64)..."
HUB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BUILD_HUB="docker buildx build \
    --platform linux/amd64 \
    -f ${HUB_DIR}/Dockerfile.hub \
    -t ${HUB_IMAGE}"

if [ "${PUSH}" = "true" ]; then
    BUILD_HUB="${BUILD_HUB} --push"
else
    BUILD_HUB="${BUILD_HUB} --load"
fi

${BUILD_HUB} "${HUB_DIR}"

echo ""
echo "=== Build complete ==="
echo "  ${NODE_IMAGE}"
echo "  ${HUB_IMAGE}"
echo ""
echo "Verify arm64 image boots on Pi:"
echo "  docker pull ${NODE_IMAGE}"
echo "  docker run --rm ${NODE_IMAGE} python -m src.main --self-test"
