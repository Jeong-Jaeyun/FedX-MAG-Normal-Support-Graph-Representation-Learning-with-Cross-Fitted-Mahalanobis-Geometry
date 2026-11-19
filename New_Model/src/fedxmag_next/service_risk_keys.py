"""Observed source/peer service equality; never numeric neural coordinates."""

from __future__ import annotations

import numpy as np

from .prototype_guard import entity_keys

PROTOCOL_COLUMNS = ("protocol_tcp", "protocol_udp", "protocol_icmp", "protocol_other")


def service_keys(keys, protocols, port_available, scope):
    protocols, available = np.asarray(protocols), np.asarray(port_available)
    if protocols.shape != (len(keys["source"]), 4) or available.shape != (len(keys["source"]),):
        raise ValueError("Service routing metadata must align with every query row")
    if not np.isin(protocols, (0, 1)).all() or not np.all(protocols.sum(axis=1) == 1):
        raise ValueError("Service routing requires the actual observed protocol category")
    if not np.isin(available, (0, 1)).all() or len(keys["port"]) != len(available):
        raise ValueError("Service port availability and destination ports differ")
    base = entity_keys(keys, scope)
    code = np.argmax(protocols, axis=1)
    return [
        (value if isinstance(value, tuple) else (value,)) + (int(port), int(proto), int(seen))
        for value, port, proto, seen in zip(base, keys["port"], code, available, strict=True)
    ]
