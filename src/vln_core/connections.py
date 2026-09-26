"""Directed connection memory with explicit observation and verification gates."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from heapq import heappop, heappush


class ConnectionState(str, Enum):
    CANDIDATE = "candidate"
    INDICATED = "indicated"
    TRAVERSED = "traversed"
    VERIFIED = "verified"
    BLOCKED = "blocked"


@dataclass
class DirectedConnection:
    edge_id: str
    source: str
    destination: str | None = None
    required_capability: str = "walk"
    cost: float = 1.0
    state: ConnectionState = ConnectionState.CANDIDATE
    evidence: dict[str, str] = field(default_factory=dict)


class ConnectionMemory:
    """Only independently verified directed edges are reusable routes.

    A sighting and sign can guide exploration, but cannot create a traversable
    route. Traversal records the outcome; a distinct arrival-identity source
    is still needed before the edge becomes verified.
    """

    def __init__(self) -> None:
        self.edges: dict[str, DirectedConnection] = {}

    def sighting(self, edge_id: str, source: str, evidence_ref: str) -> DirectedConnection:
        if not all((edge_id, source, evidence_ref)) or edge_id in self.edges:
            raise ValueError("new sighting needs a unique id, source and evidence")
        edge = DirectedConnection(edge_id=edge_id, source=source, evidence={"sighting": evidence_ref})
        self.edges[edge_id] = edge
        return edge

    def indication(self, edge_id: str, destination: str, evidence_ref: str,
                   capability: str = "walk", cost: float = 1.0) -> None:
        edge = self.edges[edge_id]
        if edge.state not in {ConnectionState.CANDIDATE, ConnectionState.INDICATED}:
            raise ValueError("indication cannot overwrite a used or blocked connection")
        if not all((destination, evidence_ref, capability)) or destination == edge.source or cost <= 0:
            raise ValueError("invalid directed indication")
        edge.destination = destination
        edge.required_capability = capability
        edge.cost = cost
        edge.evidence["indication"] = evidence_ref
        edge.state = ConnectionState.INDICATED

    def traversed(self, edge_id: str, arrival: str, evidence_ref: str) -> None:
        edge = self.edges[edge_id]
        if edge.state not in {ConnectionState.CANDIDATE, ConnectionState.INDICATED}:
            raise ValueError("connection is not eligible for a new traversal")
        if not arrival or arrival == edge.source or not evidence_ref:
            raise ValueError("invalid traversal evidence")
        # A sign may have been wrong. Record the actual endpoint for review.
        edge.destination = arrival
        edge.evidence["traversal"] = evidence_ref
        edge.state = ConnectionState.TRAVERSED

    def verified(self, edge_id: str, arrival: str, identity_evidence_ref: str) -> None:
        edge = self.edges[edge_id]
        if edge.state != ConnectionState.TRAVERSED or arrival != edge.destination:
            raise ValueError("verify the observed arrival of a traversed edge")
        if not identity_evidence_ref or identity_evidence_ref in edge.evidence.values():
            raise ValueError("arrival identity needs distinct evidence")
        edge.evidence["arrival_identity"] = identity_evidence_ref
        edge.state = ConnectionState.VERIFIED

    def blocked(self, edge_id: str, evidence_ref: str) -> None:
        if not evidence_ref:
            raise ValueError("blocking needs evidence")
        edge = self.edges[edge_id]
        edge.evidence["blocked"] = evidence_ref
        edge.state = ConnectionState.BLOCKED

    def verified_route(self, source: str, goal: str, capabilities: set[str]) -> list[str] | None:
        if source == goal:
            return []
        adjacency: dict[str, list[DirectedConnection]] = {}
        for edge in self.edges.values():
            if edge.state == ConnectionState.VERIFIED and edge.required_capability in capabilities:
                adjacency.setdefault(edge.source, []).append(edge)
        queue: list[tuple[float, str, list[str]]] = [(0.0, source, [])]
        best: dict[str, float] = {source: 0.0}
        while queue:
            cost, node, route = heappop(queue)
            if node == goal:
                return route
            if cost > best[node]:
                continue
            for edge in adjacency.get(node, []):
                assert edge.destination is not None
                updated = cost + edge.cost
                if updated < best.get(edge.destination, float("inf")):
                    best[edge.destination] = updated
                    heappush(queue, (updated, edge.destination, route + [edge.edge_id]))
        return None
