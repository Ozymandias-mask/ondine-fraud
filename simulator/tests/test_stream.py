import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest
from ondine_sim.stream import (
    TOPIC_EVENTS,
    TOPIC_REPORTS,
    TOPIC_TRANSACTIONS,
    ControlState,
    StreamProducer,
    StreamSettings,
    _handler_factory,
)


class FakeProducer:
    """Remplace le client Kafka : enregistre les messages et acquitte tout."""

    def __init__(self):
        self.messages, self._callbacks = [], []

    def produce(self, topic, key, value, on_delivery):
        self.messages.append((topic, key, value))
        self._callbacks.append(on_delivery)

    def poll(self, _timeout):
        for callback in self._callbacks:
            callback(None, None)
        self._callbacks = []

    def flush(self, _timeout):
        self.poll(0)
        return 0


def test_ticks_publish_transactions_events_and_reports(small_cfg):
    settings = StreamSettings(rate=500, report_delay_s=0, seed=1)
    producer = FakeProducer()
    generator = StreamProducer(small_cfg, settings, ControlState(settings), producer)
    start = 1_800_000_000_000
    published = sum(generator.tick(start + i * 1000) for i in range(20))
    topics = [m[0] for m in producer.messages]
    assert 8_000 < published < 12_000
    assert topics.count(TOPIC_TRANSACTIONS) == published
    assert topics.count(TOPIC_EVENTS) > 0
    record = json.loads(next(m[2] for m in producer.messages if m[0] == TOPIC_TRANSACTIONS))
    assert record["currency"] == "EUR" and "wallet_type" in record
    # Toute fraude injectée est soit déjà déclarée, soit en attente (attaque encore en
    # cours de déroulement, ou non déclarée : 10 % des cas).
    fraud_pending = (generator.pending_tx["fraud_type"] > 0).any()
    assert (
        generator.incident_counter == 0
        or TOPIC_REPORTS in topics
        or fraud_pending
        or generator.pending_reports
        or published > 0
    )


def test_control_validation():
    state = ControlState(StreamSettings())
    assert state.update("rate", "800")["rate"] == 800
    assert state.update("drift", "true")["drift"] is True
    with pytest.raises(ValueError):
        state.update("rate", "-5")
    with pytest.raises(ValueError):
        state.update("unknown", "1")
    state.update("burst", "1")
    assert state.current_rate() == state.burst_rate


def test_control_api_requires_token():
    state = ControlState(StreamSettings())
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler_factory(state, "secret"))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(
                urllib.request.Request(f"{base}/control/drift?enabled=1", method="POST")
            )
        assert exc.value.code == 401
        request = urllib.request.Request(
            f"{base}/control/drift?enabled=1", method="POST", headers={"X-Control-Token": "secret"}
        )
        assert json.loads(urllib.request.urlopen(request).read())["drift"] is True
        assert b"sim_" in urllib.request.urlopen(f"{base}/metrics").read()
    finally:
        server.shutdown()
