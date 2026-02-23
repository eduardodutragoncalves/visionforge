CREATE TABLE IF NOT EXISTS json_documents (
    path TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL,
    schema_name TEXT,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_json_documents_updated_at
ON json_documents(updated_at);
