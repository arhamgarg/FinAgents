from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pyotp

from finagents.config import Settings

if TYPE_CHECKING:
    from SmartApi.smartConnect import SmartConnect as _SmartConnectType

SmartConnect = None  # type: ignore[assignment]


class AngelOneAuthError(Exception):
    pass


class SessionManager:

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = asyncio.Lock()
        self._obj: _SmartConnectType | None = None

    async def get_session(self) -> _SmartConnectType:
        async with self._lock:
            if self._obj is None:
                await self._authenticate()
            assert self._obj is not None
            return self._obj

    async def _authenticate(self) -> None:
        if not all(
            [
                self._settings.angelone_api_key,
                self._settings.angelone_client_id,
                self._settings.angelone_pin,
                self._settings.angelone_totp_secret,
            ]
        ):
            raise AngelOneAuthError(
                "Angel One credentials missing. Set ANGELONE_API_KEY, "
                "ANGELONE_CLIENT_ID, ANGELONE_PIN, ANGELONE_TOTP_SECRET in.env"
            )

        global SmartConnect
        if SmartConnect is None:
            from SmartApi.smartConnect import SmartConnect as _RealSmartConnect

            SmartConnect = _RealSmartConnect

        assert self._settings.angelone_totp_secret is not None
        totp_now = pyotp.TOTP(self._settings.angelone_totp_secret).now()
        obj = SmartConnect(api_key=self._settings.angelone_api_key)
        response = obj.generateSession(
            clientCode=self._settings.angelone_client_id,
            password=self._settings.angelone_pin,
            totp=totp_now,
        )
        if not response.get("status"):
            raise AngelOneAuthError(
                f"generateSession failed: errorcode={response.get('errorcode')!r} "
                f"message={response.get('message')!r}"
            )

        obj.session_expiry_hook = self._reauth_sync_shim
        self._obj = obj

    def _reauth_sync_shim(self) -> None:
        raise Exception("TokenException: session expired, re-auth required")

    async def _reauthenticate(self) -> None:
        async with self._lock:
            self._obj = None
            await self._authenticate()

    async def call_with_retry(
        self,
        method_name: str,
        *args,
        max_attempts: int = 4,
        **kwargs,
    ):
        attempt = 0
        reauth_attempted = False

        while True:
            session = await self.get_session()
            method = getattr(session, method_name)

            try:
                response = method(*args, **kwargs)
            except Exception as exc:
                if "TokenException" in str(exc) and not reauth_attempted:
                    reauth_attempted = True
                    await self._reauthenticate()
                    continue
                raise

            errorcode = response.get("errorcode", "") if isinstance(response, dict) else ""

            if errorcode == "AB1010" and not reauth_attempted:
                reauth_attempted = True
                await self._reauthenticate()
                continue

            if errorcode == "AB1050":
                raise AngelOneAuthError(f"Invalid TOTP (AB1050): {response.get('message')!r}")

            if errorcode == "AB1004":
                if attempt >= max_attempts - 1:
                    raise AngelOneAuthError(
                        f"AB1004 exceeded max attempts ({max_attempts}): "
                        f"{response.get('message')!r}"
                    )
                await asyncio.sleep(2**attempt)
                attempt += 1
                continue

            if not response.get("status", True):
                raise AngelOneAuthError(
                    f"Angel One error: errorcode={errorcode!r} message={response.get('message')!r}"
                )

            return response



_instance: SessionManager | None = None


def get_session_manager(settings: Settings) -> SessionManager:
    global _instance
    if _instance is None:
        _instance = SessionManager(settings)
    return _instance


def reset_session_manager() -> None:
    global _instance
    _instance = None
