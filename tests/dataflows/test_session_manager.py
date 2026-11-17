from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from finagents.config import Settings
from finagents.dataflows.session_manager import (
    AngelOneAuthError,
    SessionManager,
    get_session_manager,
    reset_session_manager,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    reset_session_manager()
    yield
    reset_session_manager()


@pytest.fixture
def angelone_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("ANGELONE_API_KEY", "test-api-key")
    monkeypatch.setenv("ANGELONE_CLIENT_ID", "TEST01")
    monkeypatch.setenv("ANGELONE_PIN", "1234")
    monkeypatch.setenv("ANGELONE_TOTP_SECRET", "JBSWY3DPEHPK3PXP")
    return Settings(openrouter_api_key="fake")


@pytest.mark.asyncio
async def test_get_session_authenticates_once(mock_smart_connect, angelone_settings):
    with patch("finagents.dataflows.session_manager.pyotp") as mock_pyotp:
        mock_totp_instance = MagicMock()
        mock_totp_instance.now.return_value = "123456"
        mock_pyotp.TOTP.return_value = mock_totp_instance

        manager = SessionManager(angelone_settings)
        session = await manager.get_session()

        assert session is mock_smart_connect
        mock_smart_connect.generateSession.assert_called_once()
        mock_pyotp.TOTP.assert_called_once_with("JBSWY3DPEHPK3PXP")
        mock_totp_instance.now.assert_called_once()


@pytest.mark.asyncio
async def test_missing_credentials_raises_auth_error(clean_env):
    settings = Settings(openrouter_api_key="fake")
    manager = SessionManager(settings)
    with pytest.raises(AngelOneAuthError, match="Angel One credentials missing"):
        await manager.get_session()


@pytest.mark.asyncio
async def test_reauth_on_token_exception(mock_smart_connect, angelone_settings):
    success_response = {"status": True, "data": [["2024-01-01", 100, 110, 90, 105, 50000]]}
    mock_smart_connect.getCandleData.side_effect = [
        Exception("Something TokenException happened"),
        success_response,
    ]

    manager = SessionManager(angelone_settings)
    result = await manager.call_with_retry("getCandleData", {"exchange": "NSE"})

    assert result == success_response
    assert mock_smart_connect.generateSession.call_count == 2


@pytest.mark.asyncio
async def test_reauth_on_ab1010(mock_smart_connect, angelone_settings):
    ab1010_response = {
        "status": False,
        "message": "Invalid Session or Session Expired Please Re-Login",
        "errorcode": "AB1010",
        "data": None,
    }
    success_response = {"status": True, "data": []}
    mock_smart_connect.getCandleData.side_effect = [ab1010_response, success_response]

    manager = SessionManager(angelone_settings)
    result = await manager.call_with_retry("getCandleData", {"exchange": "NSE"})

    assert result == success_response
    assert mock_smart_connect.generateSession.call_count == 2


@pytest.mark.asyncio
async def test_ab1004_retries_not_reauth(mock_smart_connect, angelone_settings, monkeypatch):
    ab1004_response = {
        "status": False,
        "message": "Something Went Wrong",
        "errorcode": "AB1004",
        "data": None,
    }
    mock_smart_connect.getCandleData.return_value = ab1004_response

    sleep_calls = []

    async def mock_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr("finagents.dataflows.session_manager.asyncio.sleep", mock_sleep)

    manager = SessionManager(angelone_settings)
    with pytest.raises(AngelOneAuthError, match="AB1004"):
        await manager.call_with_retry("getCandleData", {"exchange": "NSE"}, max_attempts=4)

    assert mock_smart_connect.generateSession.call_count == 1
    assert sleep_calls == [1, 2, 4]


@pytest.mark.asyncio
async def test_ab1050_no_retry(mock_smart_connect, angelone_settings):
    ab1050_response = {
        "status": False,
        "message": "Invalid TOTP",
        "errorcode": "AB1050",
        "data": None,
    }
    mock_smart_connect.getCandleData.return_value = ab1050_response

    manager = SessionManager(angelone_settings)
    with pytest.raises(AngelOneAuthError, match="AB1050"):
        await manager.call_with_retry("getCandleData", {"exchange": "NSE"})

    assert mock_smart_connect.getCandleData.call_count == 1
    assert mock_smart_connect.generateSession.call_count == 1


@pytest.mark.asyncio
async def test_singleton_lock_serializes_first_auth(mock_smart_connect, angelone_settings):
    manager = SessionManager(angelone_settings)

    session_a, session_b = await asyncio.gather(
        manager.get_session(),
        manager.get_session(),
    )

    assert session_a is mock_smart_connect
    assert session_b is mock_smart_connect
    assert mock_smart_connect.generateSession.call_count == 1


@pytest.mark.asyncio
async def test_fresh_totp_on_reauthenticate(mock_smart_connect, angelone_settings):
    with patch("finagents.dataflows.session_manager.pyotp") as mock_pyotp:
        mock_totp_instance = MagicMock()
        mock_totp_instance.now.return_value = "123456"
        mock_pyotp.TOTP.return_value = mock_totp_instance

        ab1010_response = {
            "status": False,
            "errorcode": "AB1010",
            "message": "Session expired",
            "data": None,
        }
        success_response = {"status": True, "data": []}
        mock_smart_connect.getCandleData.side_effect = [ab1010_response, success_response]

        manager = SessionManager(angelone_settings)
        await manager.call_with_retry("getCandleData", {"exchange": "NSE"})

        assert mock_pyotp.TOTP.call_count == 2
        assert mock_totp_instance.now.call_count == 2


@pytest.mark.asyncio
async def test_credential_not_in_error_message(clean_env, monkeypatch):
    monkeypatch.setenv("ANGELONE_API_KEY", "test-api-key")
    monkeypatch.setenv("ANGELONE_CLIENT_ID", "TEST01")
    monkeypatch.setenv("ANGELONE_PIN", "super-secret-pin-1234")
    monkeypatch.setenv("ANGELONE_TOTP_SECRET", "JBSWY3DPEHPK3PXP")
    settings = Settings(openrouter_api_key="fake")

    with patch("finagents.dataflows.session_manager.SmartConnect") as mock_cls:
        instance = MagicMock()
        instance.generateSession.return_value = {
            "status": False,
            "message": "Auth failed",
            "errorcode": "AB1001",
        }
        mock_cls.return_value = instance

        manager = SessionManager(settings)
        with pytest.raises(AngelOneAuthError) as exc_info:
            await manager.get_session()

        error_msg = str(exc_info.value)
        assert "super-secret-pin-1234" not in error_msg
        assert "JBSWY3DPEHPK3PXP" not in error_msg


def test_get_session_manager_singleton(angelone_settings):
    mgr1 = get_session_manager(angelone_settings)
    mgr2 = get_session_manager(angelone_settings)
    assert mgr1 is mgr2


def test_reset_session_manager(angelone_settings):
    mgr1 = get_session_manager(angelone_settings)
    reset_session_manager()
    mgr2 = get_session_manager(angelone_settings)
    assert mgr1 is not mgr2


def test_import_emits_no_smartconnect_log() -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", "from finagents.dataflows import session_manager"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"Import failed (stdout={result.stdout!r}, stderr={result.stderr!r})"
    )
    assert "smartConnect:78" not in result.stderr, (
        f"SmartConnect import-time ERROR log leaked to stderr: {result.stderr!r}"
    )
    assert "Exception while retriving IP" not in result.stderr, (
        f"SmartConnect IP-lookup error leaked to stderr: {result.stderr!r}"
    )
    assert "SmartApi" not in result.stderr and "smartConnect" not in result.stderr, (
        f"SmartApi import-time output leaked to stderr: {result.stderr!r}"
    )
