"""CIC-IDS2017 captures (Monday, Thursday, Friday) as TON_IoT-format connection tables.

The pcapng captures are converted without labels by :mod:`fedxmag.pcap_conn`. Labels
come only from the attack schedule published with the dataset (UNB CIC, "Intrusion
Detection Evaluation Dataset (CIC-IDS2017)"). A record is an attack record of an entry
if one endpoint is an attacker address and the other a victim address of that entry
and its start second falls in the entry's interval, from the first listed minute to
the end of the last. Inside the capture, inbound attacks appear from the firewall's
inner address 172.16.0.1 (the published NAT path), so it is an attacker address of the
inbound entries. Schedule times are local (ADT, UTC-3); the captures start at
08:55-08:59 local, which fixes the offset. Monday is benign by the dataset definition.

A Thursday or Friday record that is not an attack record but touches attack
infrastructure (an attacker address of the day or the NAT address, which carries no
Monday traffic) has unknown status: the layout census showed botnet beacons to the
attacker until 17:00, past the published 10:02-11:02, and unscheduled NAT traffic.
Such records are removed from the labelled tables and counted; they are neither
normal nor attack.
"""

from __future__ import annotations

import csv
import ipaddress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

CAPTURE_OFFSET_SECONDS = -3 * 3600
SESSION_PREFIX = "cic_ids2017"
DAYS = {"Monday": "2017-07-03", "Thursday": "2017-07-06", "Friday": "2017-07-07"}
NAT = "172.16.0.1"
KALI = "205.174.165.73"
WEB_SERVER = frozenset({"192.168.10.50", "205.174.165.68"})
VICTIM_LAN = ipaddress.ip_network("192.168.10.0/24")


@dataclass(frozen=True)
class Entry:
    kind: str
    day: str
    start: str  # local HH:MM, first minute
    end: str  # local HH:MM, last minute (inclusive)
    attackers: frozenset[str]
    victims: frozenset[str] | None  # None: every other victim-LAN address
    bounds: tuple[int, int] = field(init=False)  # UTC epoch seconds, end exclusive

    def __post_init__(self) -> None:
        def seconds(clock: str) -> int:
            local = datetime.strptime(f"{self.day} {clock}", "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
            return int(local.timestamp()) - CAPTURE_OFFSET_SECONDS

        object.__setattr__(self, "bounds", (seconds(self.start), seconds(self.end) + 60))

    def matches(self, second: int, source: str, destination: str) -> bool:
        first, last = self.bounds
        if not first <= second < last:
            return False
        for attacker, victim in ((source, destination), (destination, source)):
            if attacker not in self.attackers:
                continue
            if self.victims is None:
                if victim != attacker and _in_lan(victim):
                    return True
            elif victim in self.victims:
                return True
        return False


def _in_lan(address: str) -> bool:
    try:
        return ipaddress.ip_address(address) in VICTIM_LAN
    except ValueError:
        return False


INBOUND = frozenset({NAT, KALI})
THURSDAY, FRIDAY = DAYS["Thursday"], DAYS["Friday"]
SCHEDULE = (
    Entry("web_brute_force", THURSDAY, "09:20", "10:00", INBOUND, WEB_SERVER),
    Entry("web_xss", THURSDAY, "10:15", "10:35", INBOUND, WEB_SERVER),
    Entry("web_sql_injection", THURSDAY, "10:40", "10:42", INBOUND, WEB_SERVER),
    Entry("infiltration", THURSDAY, "14:19", "14:21", INBOUND, frozenset({"192.168.10.8"})),
    Entry("infiltration", THURSDAY, "14:33", "14:35", INBOUND, frozenset({"192.168.10.8"})),
    Entry("infiltration", THURSDAY, "14:53", "15:00", INBOUND, frozenset({"192.168.10.25"})),
    Entry("infiltration", THURSDAY, "15:04", "15:45", INBOUND, frozenset({"192.168.10.8"})),
    # Second step: the infected Vista host scans every other victim-LAN address.
    Entry("infiltration", THURSDAY, "15:04", "15:45", frozenset({"192.168.10.8"}), None),
    Entry("botnet", FRIDAY, "10:02", "11:02", frozenset({KALI}), frozenset({
        "192.168.10.15", "192.168.10.9", "192.168.10.14", "192.168.10.5", "192.168.10.8"})),
    Entry("portscan", FRIDAY, "13:55", "15:29", INBOUND, WEB_SERVER),
    Entry("ddos", FRIDAY, "15:56", "16:16", frozenset(
        {NAT, "205.174.165.69", "205.174.165.70", "205.174.165.71"}), WEB_SERVER),
)
ATTACK_INFRASTRUCTURE = {
    THURSDAY: frozenset({KALI, NAT}),
    FRIDAY: frozenset({KALI, NAT, "205.174.165.69", "205.174.165.70", "205.174.165.71"}),
}


def capture_of(timestamp_ns: int) -> str:
    seconds = timestamp_ns // 1_000_000_000 + CAPTURE_OFFSET_SECONDS
    return datetime.fromtimestamp(seconds, UTC).strftime("%Y-%m-%d")


def session_key(capture: str) -> str:
    return f"{SESSION_PREFIX}:{capture}"


def label(second: int, source: str, destination: str) -> tuple[str | None, bool]:
    """(attack kind or None, unknown status) of a record starting at this UTC second."""

    day = capture_of(second * 1_000_000_000)
    for entry in SCHEDULE:
        if entry.day == day and entry.matches(second, source, destination):
            return entry.kind, False
    infrastructure = ATTACK_INFRASTRUCTURE.get(day, frozenset())
    return None, source in infrastructure or destination in infrastructure


def write_labelled(conn_table: str | Path, destination: str | Path) -> dict[str, int]:
    """Copy a converted table, append ``label`` and ``type``, drop unknown-status rows."""

    counts: dict[str, int] = {}
    with Path(conn_table).open(encoding="utf-8", newline="") as source, Path(
        destination
    ).open("x", encoding="utf-8", newline="") as target:
        reader, writer = csv.reader(source), csv.writer(target)
        fields = next(reader)
        index = {name: fields.index(name) for name in ("ts", "src_ip", "dst_ip")}
        writer.writerow([*fields, "label", "type"])
        for row in reader:
            kind, unknown = label(int(row[index["ts"]]), row[index["src_ip"]],
                                  row[index["dst_ip"]])
            key = kind or ("unknown_dropped" if unknown else "normal")
            counts[key] = counts.get(key, 0) + 1
            if not unknown:
                writer.writerow([*row, "1" if kind else "0", kind or "normal"])
    return counts
