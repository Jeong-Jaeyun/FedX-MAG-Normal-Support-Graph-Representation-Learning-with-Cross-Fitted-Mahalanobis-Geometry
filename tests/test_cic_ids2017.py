import csv
from datetime import UTC, datetime

from fedxmag.cic_ids2017 import capture_of, label, write_labelled


def utc_second(day, clock):
    """UTC epoch second of a local (ADT, UTC-3) clock time."""

    local = datetime.strptime(f"{day} {clock}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    return int(local.timestamp()) + 3 * 3600


def test_capture_is_the_local_day():
    assert capture_of(utc_second("2017-07-06", "08:58:00") * 10**9) == "2017-07-06"
    assert capture_of(utc_second("2017-07-06", "23:59:59") * 10**9) == "2017-07-06"
    assert capture_of(utc_second("2017-07-07", "00:00:00") * 10**9) == "2017-07-07"


def test_schedule_labels_by_endpoints_and_interval():
    thursday, friday = "2017-07-06", "2017-07-07"
    web = ("172.16.0.1", "192.168.10.50")
    assert label(utc_second(thursday, "09:19:59"), *web) == (None, True)
    assert label(utc_second(thursday, "09:20:00"), *web) == ("web_brute_force", False)
    assert label(utc_second(thursday, "10:00:59"), *reversed(web)) == ("web_brute_force", False)
    assert label(utc_second(thursday, "10:01:00"), *web) == (None, True)  # NAT, unscheduled
    assert label(utc_second(thursday, "10:01:00"), "192.168.10.3", "192.168.10.50") == (
        None, False)
    assert label(utc_second(thursday, "10:41:00"), *web) == ("web_sql_injection", False)
    assert label(utc_second(thursday, "15:10:00"), "192.168.10.8", "192.168.10.12") == (
        "infiltration", False)
    assert label(utc_second(thursday, "15:10:00"), "192.168.10.8", "8.8.8.8") == (None, False)
    bot = ("192.168.10.5", "205.174.165.73")
    assert label(utc_second(friday, "10:30:00"), *bot) == ("botnet", False)
    assert label(utc_second(friday, "12:00:00"), *bot) == (None, True)
    assert label(utc_second(friday, "16:00:00"), "205.174.165.70", "192.168.10.50") == (
        "ddos", False)
    assert label(utc_second("2017-07-03", "10:30:00"), *bot) == (None, False)


def test_write_labelled_appends_label_and_type(tmp_path):
    second = utc_second("2017-07-07", "14:00:00")
    with (tmp_path / "conn.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ts", "src_ip", "dst_ip", "proto"])
        writer.writerow([second, "172.16.0.1", "192.168.10.50", "tcp"])
        writer.writerow([second, "192.168.10.3", "192.168.10.50", "udp"])
        writer.writerow([second, "192.168.10.9", "205.174.165.73", "tcp"])  # beacon, unscheduled
    counts = write_labelled(tmp_path / "conn.csv", tmp_path / "labelled.csv")
    with (tmp_path / "labelled.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    assert rows[0] == ["ts", "src_ip", "dst_ip", "proto", "label", "type"]
    assert [row[-2:] for row in rows[1:]] == [["1", "portscan"], ["0", "normal"]]
    assert counts == {"portscan": 1, "normal": 1, "unknown_dropped": 1}
