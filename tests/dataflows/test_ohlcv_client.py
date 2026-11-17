from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from finagents.clock import IST
from finagents.dataflows.scrip_master import InstrumentNotFoundError
from finagents.dataflows.session_manager import AngelOneAuthError

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "angelone"


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text())


@pytest.fixture(autouse=True)
def reset_session(monkeypatch):
    import finagents.dataflows.session_manager as sm_mod

    monkeypatch.setattr(sm_mod, "_instance", None)
    yield
    monkeypatch.setattr(sm_mod, "_instance", None)


@pytest.fixture
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db = tmp_path / "test_ohlcv.db"
    monkeypatch.setattr("finagents.dataflows.cache.CACHE_DB_PATH", db)
    return db


@pytest.fixture
def angel_settings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ANGELONE_API_KEY", "test-api-key")
    monkeypatch.setenv("ANGELONE_CLIENT_ID", "TEST01")
    monkeypatch.setenv("ANGELONE_PIN", "1234")
    monkeypatch.setenv("ANGELONE_TOTP_SECRET", "JBSWY3DPEHPK3PXP")
    from finagents.config import Settings

    return Settings(openrouter_api_key="fake")


@pytest.mark.asyncio
async def test_ohlcv_daily_single_request(
    mock_smart_connect: MagicMock,
    isolated_cache: Path,
    angel_settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daily_fixture = _load_fixture("ohlcv_daily.json")
    mock_smart_connect.getCandleData.return_value = daily_fixture

    monkeypatch.setattr(
        "finagents.dataflows.ohlcv_client.scrip_master.resolve_token",
        AsyncMock(return_value="2885"),
    )

    from finagents.dataflows.ohlcv_client import get_ohlcv

    start = datetime(2024, 1, 1, 9, 15, tzinfo=IST)
    end = datetime(2024, 1, 12, 15, 30, tzinfo=IST)
    result = await get_ohlcv("RELIANCE", start, end, interval="ONE_DAY", settings=angel_settings)

    assert result.status == "ok"
    assert result.source == "angelone"
    assert result.data is not None
    assert len(result.data) == 10
    assert mock_smart_connect.getCandleData.call_count == 1


@pytest.mark.asyncio
async def test_ohlcv_intraday_chunks(
    mock_smart_connect: MagicMock,
    isolated_cache: Path,
    angel_settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page1 = _load_fixture("ohlcv_intraday_page1.json")
    page2 = _load_fixture("ohlcv_intraday_page2.json")
    mock_smart_connect.getCandleData.side_effect = [page1, page2]

    monkeypatch.setattr(
        "finagents.dataflows.ohlcv_client.scrip_master.resolve_token",
        AsyncMock(return_value="2885"),
    )

    from finagents.dataflows.ohlcv_client import get_ohlcv

    start = datetime(2024, 1, 1, 9, 15, tzinfo=IST)
    end = datetime(2024, 1, 12, 15, 30, tzinfo=IST)
    result = await get_ohlcv(
        "RELIANCE", start, end, interval="FIVE_MINUTE", settings=angel_settings
    )

    assert result.status == "ok"
    assert result.data is not None
    assert len(result.data) == 6
    assert mock_smart_connect.getCandleData.call_count == 2
    ts = result.data["timestamp"].tolist()
    assert ts == sorted(ts)


@pytest.mark.asyncio
async def test_ohlcv_cache_hit(
    mock_smart_connect: MagicMock,
    isolated_cache: Path,
    angel_settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daily_fixture = _load_fixture("ohlcv_daily.json")
    mock_smart_connect.getCandleData.return_value = daily_fixture

    monkeypatch.setattr(
        "finagents.dataflows.ohlcv_client.scrip_master.resolve_token",
        AsyncMock(return_value="2885"),
    )

    from finagents.dataflows.ohlcv_client import get_ohlcv

    start = datetime(2024, 1, 1, 9, 15, tzinfo=IST)
    end = datetime(2024, 1, 12, 15, 30, tzinfo=IST)

    r1 = await get_ohlcv("RELIANCE", start, end, interval="ONE_DAY", settings=angel_settings)
    assert r1.status == "ok"

    r2 = await get_ohlcv("RELIANCE", start, end, interval="ONE_DAY", settings=angel_settings)
    assert r2.status == "ok"
    assert r2.data is not None
    assert len(r2.data) == 10

    assert mock_smart_connect.getCandleData.call_count == 1


@pytest.mark.asyncio
async def test_instrument_not_found(
    mock_smart_connect: MagicMock,
    isolated_cache: Path,
    angel_settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "finagents.dataflows.ohlcv_client.scrip_master.resolve_token",
        AsyncMock(side_effect=InstrumentNotFoundError("symbol='BADTICKER' exchange='NSE'")),
    )

    from finagents.dataflows.ohlcv_client import get_ohlcv

    start = datetime(2024, 1, 1, 9, 15, tzinfo=IST)
    end = datetime(2024, 1, 12, 15, 30, tzinfo=IST)
    result = await get_ohlcv("BADTICKER", start, end, interval="ONE_DAY", settings=angel_settings)

    assert result.status == "error"
    assert result.source == "angelone"
    assert result.error is not None
    assert "Instrument not found" in result.error
    assert mock_smart_connect.getCandleData.call_count == 0


@pytest.mark.asyncio
async def test_auth_error_wrapped(
    mock_smart_connect: MagicMock,
    isolated_cache: Path,
    angel_settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "finagents.dataflows.ohlcv_client.scrip_master.resolve_token",
        AsyncMock(return_value="2885"),
    )
    mock_smart_connect.getCandleData.side_effect = AngelOneAuthError("AB1050 invalid TOTP")

    from finagents.dataflows.ohlcv_client import get_ohlcv

    start = datetime(2024, 1, 1, 9, 15, tzinfo=IST)
    end = datetime(2024, 1, 12, 15, 30, tzinfo=IST)
    result = await get_ohlcv("RELIANCE", start, end, interval="ONE_DAY", settings=angel_settings)

    assert result.status == "error"
    assert result.source == "angelone"
    assert result.error is not None


def test_chunk_math() -> None:
    from finagents.dataflows.ohlcv_client import _split_date_range

    start = datetime(2024, 1, 1, tzinfo=IST)
    twelve_days_end = datetime(2024, 1, 12, tzinfo=IST)
    one_k_days_end = datetime(2026, 9, 27, tzinfo=IST)
    forty_five_days_end = datetime(2024, 2, 15, tzinfo=IST)

    five_min_chunks = _split_date_range(start, twelve_days_end, "FIVE_MINUTE")
    assert len(five_min_chunks) == 2

    daily_chunks = _split_date_range(start, one_k_days_end, "ONE_DAY")
    assert len(daily_chunks) == 1

    thirty_min_chunks = _split_date_range(start, forty_five_days_end, "THIRTY_MINUTE")
    assert len(thirty_min_chunks) == 2


@pytest.mark.asyncio
async def test_dataframe_columns(
    mock_smart_connect: MagicMock,
    isolated_cache: Path,
    angel_settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daily_fixture = _load_fixture("ohlcv_daily.json")
    mock_smart_connect.getCandleData.return_value = daily_fixture

    monkeypatch.setattr(
        "finagents.dataflows.ohlcv_client.scrip_master.resolve_token",
        AsyncMock(return_value="2885"),
    )

    from finagents.dataflows.ohlcv_client import get_ohlcv

    start = datetime(2024, 1, 1, 9, 15, tzinfo=IST)
    end = datetime(2024, 1, 12, 15, 30, tzinfo=IST)
    result = await get_ohlcv("RELIANCE", start, end, interval="ONE_DAY", settings=angel_settings)

    assert result.status == "ok"
    assert result.data is not None
    assert list(result.data.columns) == ["timestamp", "open", "high", "low", "close", "volume"]


@pytest.mark.asyncio
async def test_empty_response_is_empty_status(
    mock_smart_connect: MagicMock,
    isolated_cache: Path,
    angel_settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_smart_connect.getCandleData.return_value = {"status": True, "data": []}

    monkeypatch.setattr(
        "finagents.dataflows.ohlcv_client.scrip_master.resolve_token",
        AsyncMock(return_value="2885"),
    )

    from finagents.dataflows.ohlcv_client import get_ohlcv

    start = datetime(2024, 1, 1, 9, 15, tzinfo=IST)
    end = datetime(2024, 1, 12, 15, 30, tzinfo=IST)
    result = await get_ohlcv("RELIANCE", start, end, interval="ONE_DAY", settings=angel_settings)

    assert result.status == "empty"
    assert result.source == "angelone"
    assert result.data is None
