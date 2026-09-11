"""src/auth.py — Authentication helpers for AEGIS Hub.

Rules (non-negotiable):
  1. AEGIS_AUTH_TOKEN is REQUIRED at boot unless AEGIS_DEV_MODE=1.
     If neither is set: sys.exit with a clear FATAL message.
  2. In DEV_MODE, authentication is skipped but a loud WARNING is logged
     on EVERY node connection (not just once at boot).
  3. Token comparison uses hmac.compare_digest to prevent timing attacks.
  4. 'privileged: true' in Docker is banned — if auth fails, fix the token.
"""
from __future__ import annotations

import hmac
import logging
import os
import sys

logger = logging.getLogger("aegis.hub.auth")


def check_boot_auth(auth_token: str | None, dev_mode: bool) -> None:
    """
    Validate auth configuration at hub boot time. Exits non-zero if unsafe.

    Args:
        auth_token: The pre-shared token (from env or CLI).
        dev_mode:   If True, auth is skipped but a warning banner is printed.

    Raises:
        SystemExit: If no token AND no dev_mode — fails fast with a FATAL msg.
    """
    if not auth_token and not dev_mode:
        sys.exit(
            "FATAL: AEGIS_AUTH_TOKEN is unset and AEGIS_DEV_MODE is not 1.\n"
            "  Set AEGIS_AUTH_TOKEN in your environment, or launch with\n"
            "  AEGIS_DEV_MODE=1 for local development without credentials.\n"
            "  Never use AEGIS_DEV_MODE=1 in production."
        )

    if dev_mode:
        logger.warning("=" * 60)
        logger.warning("AEGIS_DEV_MODE=1 ACTIVE — AUTHENTICATION BYPASSED")
        logger.warning("DO NOT USE IN PRODUCTION OR OUTSIDE ISOLATED LOCAL DEV")
        logger.warning("=" * 60)


def verify_token(provided: str | None, expected: str | None) -> bool:
    """
    Constant-time token comparison.

    Returns False if either argument is falsy (None / empty string).
    """
    if not provided or not expected:
        return False
    try:
        return hmac.compare_digest(provided, expected)
    except (TypeError, ValueError):
        return False


def warn_dev_mode_connection(node_id: str) -> None:
    """
    Log a per-connection warning when DEV_MODE is active.
    Called on every node_hello in dev mode — not just at boot.
    """
    logger.warning(
        f"AEGIS_DEV_MODE: authentication skipped for node '{node_id}'. "
        "This connection would be REJECTED in production."
    )
