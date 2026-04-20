# Archived Modules

This directory preserves modules that were retired from production at a specific commit but kept accessible for forensics (reading what the retired code looked like without reconstructing it from git history).

**Guardrail ** `financialagents/_archive/` has NO `__init__.py` and is intentionally NOT a Python package. `from financialagents._archive.sentiment import ...` raises `ModuleNotFoundError` at import time. If you need to reference archived code, read the file directly or use `git show <sha>:<original-path>`.

## Tombstone Index

One row per retired file. Format: file | last-active SHA (recovery via `git show <sha>:<original-path>`) | retirement phase + date | one-line reason | replacement pointer.



| file | last-active SHA | retirement phase + date | reason | replacement |
|------|-----------------|-------------------------|--------|-------------|
| `financialagents/agents/analysts/sentiment_analyst.py` | `e5b7337b310b301431dcb179d69fd9a391605de4` | , 2026-04-20 | post_count=0 across Wave 5 UAT; analyst produced no signal | None — 3-analyst flow (technical / fundamentals / news) is the production contract |
| `financialagents/dataflows/sentiment_client.py` | `71b960dd2ecb64665c8b84e250d1948a418c01cd` | , 2026-04-20 | Reddit 7.5% success, StockTwits discontinued, Twitter 0% — predictive value below noise floor | None — Reddit/StockTwits plumbing not reused |
| `financialagents/dataflows/sentiment_schema.py` | `71b960dd2ecb64665c8b84e250d1948a418c01cd` | , 2026-04-20 | `SentimentReport` only consumer was sentiment_client; archives with it | None — Reddit/StockTwits plumbing not reused |
| `tests/agents/analysts/test_sentiment_analyst.py` | `835afa5ee288e53dd8de6cf72c50aded15ac2146` | , 2026-04-20 | Test subject is the retired sentiment_analyst node | None — archived with subject  |
| `tests/agents/analysts/test_sentiment_analyst_confidence.py` | `09b5644fc421a1a58d248687163bd5f83a221bc1` | , 2026-04-20 | Fix B .1-05) 4 tests for sentiment confidence hybrid; sole subject is the retired analyst | None — archived with subject  |
| `tests/dataflows/test_sentiment_client.py` | `71b960dd2ecb64665c8b84e250d1948a418c01cd` | , 2026-04-20 | Test subject is the retired sentiment_client | None — archived with subject  |
| `tests/fixtures/sentiment/` (5 JSON fixtures) | `891b9a6054bad256d0bda46ed7353aec168c3aa9` | , 2026-04-20 | Fixtures consumed only by the retired sentiment tests | None — archived with consumers |

## Recovery

To view any archived file at its last-active state:

```bash
git show <last-active-sha>:<original-path>
# Example:
git show e5b7337b310b301431dcb179d69fd9a391605de4:financialagents/agents/analysts/sentiment_analyst.py
```

Or read the archived copy directly — it is a byte-identical rename of the last-active version.
