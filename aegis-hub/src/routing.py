"""src/routing.py — Mesh Audio Router (header fast-parse, echo suppression, fan-out).

Design invariants:
  1. The hub NEVER inspects the payload — only the header (bytes 0..14+L).
  2. Echo suppression: the originating node NEVER receives its own frames.
  3. Subscription filters: each connection receives only subscribed-to sources.
  4. No asyncio.gather over sends: route_binary() calls cq.enqueue() (sync),
     which is O(N) over connections with zero await points in the hot path.
  5. A slow consumer's ConnQueue absorbs its own backpressure — it cannot
     stall or delay the router loop for other connections.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Optional, Set

from .frames import parse_source_id

if TYPE_CHECKING:
    from .connqueue import ConnQueue

logger = logging.getLogger("aegis.hub.routing")


class MeshRouter:
    """
    Routes binary mesh frames from a source node to all subscribed targets.

    Targets can be other nodes (node→node audio) or dashboards.
    Both are abstracted as (ConnQueue, subscriptions) pairs so the routing
    logic is identical — the router doesn't care about connection type.
    """

    def route_binary(
        self,
        data: bytes,
        src_node_id: str,
        nodes: Dict[str, "_NodeConn"],      # node_id → NodeConn
        dashboards: Dict[object, "_DashConn"],  # ws → DashConn
    ) -> int:
        """
        Fan-out a binary frame from src_node_id to all subscribed targets.

        Args:
            data:         Raw binary frame bytes (passed through byte-identical).
            src_node_id:  Originating node ID (from the parsed header).
            nodes:        Registry of connected nodes {node_id: NodeConn}.
            dashboards:   Registry of connected dashboards {ws: DashConn}.

        Returns:
            Number of connections the frame was enqueued to.
        """
        routed = 0

        # Node→Node fan-out (echo suppressed)
        for nid, conn in nodes.items():
            if nid == src_node_id:
                continue                                   # echo suppression
            if not conn.subscribes(src_node_id):
                continue                                   # subscription filter
            conn.q.enqueue(data)                           # sync, never blocks
            routed += 1

        # Node→Dashboard fan-out
        for dash in dashboards.values():
            if not dash.subscribes(src_node_id):
                continue
            dash.q.enqueue(data)
            routed += 1

        return routed

    def route_json(
        self,
        message: str,
        src_node_id: Optional[str],
        dashboards: Dict[object, "_DashConn"],
    ) -> None:
        """
        Fan-out a JSON control message from a node to all subscribed dashboards.
        (JSON messages do not route node→node.)
        """
        for dash in dashboards.values():
            if src_node_id is None or dash.subscribes(src_node_id):
                dash.q.enqueue(message)


# ---------------------------------------------------------------------------
# Subscription helpers — mixed into NodeConn / DashConn
# ---------------------------------------------------------------------------

class SubscriptionMixin:
    """
    Mixin that gives a connection object a subscription set.

    Default: subscribes to all sources ("*").
    Call set_subscriptions(["node-a", "node-b"]) to restrict.
    Call set_subscriptions(["*"]) to reset to all.
    """

    def __init_subscription__(self) -> None:
        self._subscriptions: Optional[Set[str]] = None   # None = all (*)

    def set_subscriptions(self, node_ids: list[str]) -> None:
        if not node_ids or "*" in node_ids:
            self._subscriptions = None   # wildcard
        else:
            self._subscriptions = set(node_ids)

    def subscribes(self, source_node_id: str) -> bool:
        """True if this connection should receive frames from source_node_id."""
        return self._subscriptions is None or source_node_id in self._subscriptions

    @property
    def subscription_list(self) -> list[str]:
        return ["*"] if self._subscriptions is None else sorted(self._subscriptions)
