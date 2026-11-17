from __future__ import annotations

import re
from zoneinfo import ZoneInfo


class TestISTDates:
    def test_ist_today_str_format(self) -> None:
        from finagents.clock import ist_today_str, utc_now

        result = ist_today_str()
        assert re.match(r"^\d{4}-\d{2}-\d{2}$", result), (
            f"unexpected ist_today_str() output: {result!r}"
        )
        expected = utc_now().astimezone(ZoneInfo("Asia/Kolkata")).date().isoformat()
        assert result == expected
