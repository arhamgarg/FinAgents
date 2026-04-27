PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS _meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS signals (
    run_id TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    created_at TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('running','success','cancelled','errored')),
    action TEXT,
    confidence REAL,
    position_size_pct REAL,
    approved INTEGER,
    pm_decision TEXT,
    error_msg TEXT,
    payload_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_signals_created_at ON signals(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_signals_ticker_date ON signals(ticker, date);
CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status);

INSERT OR REPLACE INTO _meta(key, value) VALUES ('schema_version', '1');
