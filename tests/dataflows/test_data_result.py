from __future__ import annotations

import warnings

import pytest
from pydantic import ValidationError

from financialagents.config import Settings
from financialagents.dataflows.data_result import DataResult


class TestDataResult:
    def test_ok_result_with_dict_payload(self) -> None:
        result = DataResult[dict](status="ok", data={"x": 1}, source="screener_csv")
        assert result.status == "ok"
        assert result.data == {"x": 1}
        assert result.source == "screener_csv"
        assert result.stale is False
        assert result.fetched_at.tzinfo is not None
        assert result.fetched_at.utcoffset().total_seconds() == 0

    def test_empty_result_with_list_payload(self) -> None:
        result = DataResult[list](status="empty", error="REDDIT_CLIENT_ID not set", source="reddit")
        assert result.status == "empty"
        assert result.data is None
        assert result.error == "REDDIT_CLIENT_ID not set"
        assert result.source == "reddit"

    def test_invalid_status_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            DataResult.model_validate({"status": "bogus"})

    def test_missing_optional_keys_warns(self, clean_env: None) -> None:
        with pytest.warns(UserWarning, match="ANGELONE_API_KEY"):
            Settings(_env_file=None)

    def test_all_optional_keys_set_no_warnings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANGELONE_API_KEY", "test-key")
        monkeypatch.setenv("ANGELONE_CLIENT_ID", "TEST01")
        monkeypatch.setenv("ANGELONE_PIN", "1234")
        monkeypatch.setenv("ANGELONE_TOTP_SECRET", "JBSWY3DPEHPK3PXP")
        monkeypatch.setenv("GROQ_API_KEY", "test-groq-api-key-fake")
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            Settings(_env_file=None)

    def test_data_vendors_default_has_all_categories(self, clean_env: None) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            settings = Settings(_env_file=None)
        assert "ohlcv" in settings.data_vendors
        assert "fundamentals" in settings.data_vendors
        assert "news" in settings.data_vendors
        assert "sentiment" not in settings.data_vendors
