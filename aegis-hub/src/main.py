"""src/main.py — AEGIS Hub Boot Script.

Thin entry point that:
  1. Reads config from env / CLI args
  2. Runs auth fail-fast check (AEGIS_AUTH_TOKEN required unless AEGIS_DEV_MODE=1)
  3. Creates a pre-bound TCP socket with TCP_NODELAY set at the listener level
  4. Hands the socket to AegisHub.serve_forever()

Compatibility shim: `from src.main import AegisHub` still works so the
existing test_hub_optimization.py suite imports remain valid.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import socket
import ssl
import sys

# ---------------------------------------------------------------------------
# Compatibility shim — keeps existing `from src.main import AegisHub` valid
# ---------------------------------------------------------------------------
from .hub import AegisHub   # re-export: tests that do `from src.main import AegisHub` still work

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] HUB: %(message)s",
)
logger = logging.getLogger("aegis.hub.main")


def _build_ssl_context(cert_path: str, key_path: str) -> ssl.SSLContext | None:
    """Create a TLS server context. Returns None if files don't exist."""
    if not (cert_path and key_path):
        return None
    if not (os.path.exists(cert_path) and os.path.exists(key_path)):
        logger.warning(
            f"SSL cert/key not found at ({cert_path}, {key_path}) — "
            "falling back to plain ws://"
        )
        return None
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=cert_path, keyfile=key_path)
    logger.info("TLS context built from cert+key files")
    return ctx


def _make_bound_socket(host: str, port: int) -> socket.socket:
    """
    Create and bind a TCP socket with TCP_NODELAY + SO_REUSEADDR set BEFORE
    any connection is accepted. This ensures the listening socket itself
    has optimal settings, not just per-connection post-accept.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.listen(128)   # backlog: 128 concurrent half-open connections
    logger.info(
        f"Pre-bound socket: {host}:{port} "
        f"TCP_NODELAY={s.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)} "
        f"SO_REUSEADDR={s.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR)}"
    )
    return s


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project AEGIS Hub Server")
    parser.add_argument(
        "--host", default=os.getenv("AEGIS_HUB_HOST", "0.0.0.0"),
        help="Bind host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=int(os.getenv("AEGIS_HUB_PORT", 8001)),
        help="Bind port (default: 8001)"
    )
    parser.add_argument(
        "--token", default=os.getenv("AEGIS_AUTH_TOKEN"),
        help="Node authentication token"
    )
    parser.add_argument(
        "--dev", action="store_true",
        help="Run in dev mode (skip auth, loud warning per connection)"
    )
    parser.add_argument(
        "--ssl-cert", default=os.getenv("AEGIS_HUB_SSL_CERT"),
        help="Path to TLS certificate"
    )
    parser.add_argument(
        "--ssl-key", default=os.getenv("AEGIS_HUB_SSL_KEY"),
        help="Path to TLS private key"
    )
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> None:
    """Main async boot sequence."""
    dev_mode   = args.dev or os.getenv("AEGIS_DEV_MODE", "0").lower() in ("1", "true", "yes")
    auth_token = args.token

    # Fail-fast auth check
    if not auth_token and not dev_mode:
        sys.exit(
            "FATAL: AEGIS_AUTH_TOKEN is unset and AEGIS_DEV_MODE is not 1.\n"
            "  Set AEGIS_AUTH_TOKEN in your environment, or use --dev for local testing.\n"
            "  Never use --dev in production."
        )

    hub = AegisHub(
        host             = args.host,
        port             = args.port,
        auth_token       = auth_token,
        dev_mode         = dev_mode,
        ssl_cert         = args.ssl_cert,
        ssl_key          = args.ssl_key,
    )

    # Pre-bound socket with TCP_NODELAY at listener level
    try:
        sock = _make_bound_socket(args.host, args.port)
    except OSError as e:
        logger.error(f"Failed to bind {args.host}:{args.port}: {e}")
        sys.exit(1)

    logger.info(f"AEGIS Hub ready — node://{args.host}:{args.port}/node  dashboard://{args.host}:{args.port}/dashboard")
    await hub.serve_forever(sock=sock)


def main() -> None:
    args = parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
