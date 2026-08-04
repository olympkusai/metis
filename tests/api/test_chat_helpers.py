"""Regression tests for chat SSE layer helpers.

Covers the bug where `state_update["assets"]` is `dict[str, AssetState]`
(Pydantic objects) and the SSE layer tried to call `.get()` on them.
"""
from metis.api.chat import _as_dict
from metis.agent.graph import AssetState


class TestAsDict:
    def test_returns_empty_dict_for_none(self):
        assert _as_dict(None) == {}

    def test_passes_through_plain_dict(self):
        d = {"live_price": 1.23, "volume_24h": 4.5}
        assert _as_dict(d) is d  # same identity, no copy

    def test_serializes_asset_state(self):
        asset = AssetState(symbol="XRPUSDT", live_price=2.5, volume_24h=100.0)
        result = _as_dict(asset)
        assert isinstance(result, dict)
        assert result["symbol"] == "XRPUSDT"
        assert result["live_price"] == 2.5
        assert result["volume_24h"] == 100.0

    def test_supports_get_after_normalization(self):
        # Direct regression for the AttributeError on chat.py:286.
        asset = AssetState(symbol="LINKUSDT", live_price=9.45)
        normalized = _as_dict(asset)
        assert normalized.get("live_price") == 9.45
        assert normalized.get("missing_field") is None
        assert normalized.get("missing_field", "default") == "default"

    def test_returns_empty_dict_for_unknown_type(self):
        assert _as_dict(42) == {}
        assert _as_dict("string") == {}
        assert _as_dict([1, 2]) == {}

    def test_handles_assets_dict_lookup(self):
        # Mirrors the exact access pattern in chat.py event_generator.
        assets = {
            "XRPUSDT": AssetState(symbol="XRPUSDT", live_price=2.5),
            "LINKUSDT": AssetState(symbol="LINKUSDT", live_price=9.45),
        }
        primary_sym = "XRPUSDT"
        primary = _as_dict(assets.get(primary_sym))
        assert primary.get("live_price") == 2.5

        # Missing symbol should not crash.
        missing = _as_dict(assets.get("BTCUSDT"))
        assert missing == {}
        assert missing.get("live_price") is None
