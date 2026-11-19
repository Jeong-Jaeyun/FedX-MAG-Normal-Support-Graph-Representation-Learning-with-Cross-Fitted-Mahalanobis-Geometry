import numpy as np
import pandas as pd

from fedxmag_next.protocol_fields import RAW_COLUMNS, encode
from fedxmag_next.record_data import FEATURES
from fedxmag_next.record_experiment import corruption_groups
from fedxmag_next.temporal_context import CONTEXT_FEATURES


def test_protocol_shapes_are_observed_and_ignore_labels_addresses_and_literal_cert_names():
    blank = {name: "-" for name in RAW_COLUMNS}
    rows = [
        blank | {"proto": "icmp", "src_port": "5", "dst_port": "1"},
        blank
        | {
            "proto": "udp",
            "service": "dns",
            "src_port": "50000",
            "dst_port": "53",
            "dns_query": "a1.test.local",
            "dns_qtype": "1",
            "dns_rcode": "3",
            "dns_RD": "T",
            "dns_RA": "F",
            "dns_AA": "F",
            "dns_rejected": "F",
        },
        blank
        | {
            "proto": "tcp",
            "service": "ssl",
            "src_port": "443",
            "dst_port": "50000",
            "ssl_version": "TLSv1.2",
            "ssl_resumed": "F",
            "ssl_established": "T",
            "ssl_subject": "CN=one",
            "ssl_issuer": "CN=one",
        },
        blank
        | {
            "proto": "tcp",
            "service": "http",
            "src_port": "50000",
            "dst_port": "80",
            "http_method": "POST",
            "http_status_code": "403",
            "http_request_body_len": "100",
            "http_response_body_len": "10",
        },
    ]
    frame = pd.DataFrame(rows)
    names, x = encode(frame)
    index = {name: i for i, name in enumerate(names)}
    assert x.shape == (4, 94) and np.isfinite(x).all()
    assert x[0, index["control_icmp_source_5"]] == 1
    assert x[0, index["control_source_port_available"]] == 0
    assert x[1, index["payload_dns_qtype_1"]] == 1
    assert x[1, index["payload_dns_rcode_3"]] == 1
    assert x[1, index["payload_dns_query_digit_fraction"]] > 0
    assert x[2, index["payload_tls_version_tlsv12"]] == 1
    assert x[2, index["payload_tls_subject_equals_issuer"]] == 1
    assert x[3, index["payload_http_status_4xx"]] == 1
    assert x[3, index["payload_log_http_request_body_len"]] > 0
    frame["label"], frame["type"], frame["src_ip"] = 1, "attack", "10.1.2.3"
    frame.loc[2, ["ssl_subject", "ssl_issuer"]] = "CN=another"
    names2, x2 = encode(frame)
    assert names == names2 and np.array_equal(x, x2)
    groups = corruption_groups(FEATURES + names + CONTEXT_FEATURES)
    icmp_column = len(FEATURES) + index["control_icmp_source_5"]
    port_column = FEATURES.index("log_dst_port")
    assert any(icmp_column in group and port_column in group for group in groups)
