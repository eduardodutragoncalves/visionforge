CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    applied_at TEXT NOT NULL,
    checksum TEXT
);

CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    analysis_type TEXT NOT NULL DEFAULT 'classification',
    primary_variant_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_projects_created_at
ON projects(created_at);

CREATE TABLE IF NOT EXISTS project_dataset_paths (
    project_id TEXT NOT NULL,
    path TEXT NOT NULL,
    added_at TEXT,
    PRIMARY KEY (project_id, path),
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS datasets (
    project_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    root_path TEXT NOT NULL DEFAULT '',
    task TEXT NOT NULL DEFAULT 'classification',
    format TEXT NOT NULL DEFAULT 'yolo',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS dataset_sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_path TEXT NOT NULL,
    added_at TEXT,
    FOREIGN KEY (project_id) REFERENCES datasets(project_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_dataset_sources_project
ON dataset_sources(project_id);

CREATE TABLE IF NOT EXISTS dataset_classes (
    project_id TEXT NOT NULL,
    class_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    color TEXT,
    PRIMARY KEY (project_id, class_id),
    FOREIGN KEY (project_id) REFERENCES datasets(project_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_dataset_classes_project_name
ON dataset_classes(project_id, name);

CREATE TABLE IF NOT EXISTS dataset_images (
    project_id TEXT NOT NULL,
    image_id TEXT NOT NULL,
    path TEXT NOT NULL,
    split TEXT NOT NULL DEFAULT 'unassigned',
    detection_done INTEGER NOT NULL DEFAULT 0,
    is_augmented INTEGER NOT NULL DEFAULT 0,
    augmentation_source_id TEXT,
    created_at TEXT,
    updated_at TEXT,
    PRIMARY KEY (project_id, image_id),
    UNIQUE (project_id, path),
    FOREIGN KEY (project_id) REFERENCES datasets(project_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_dataset_images_project_split
ON dataset_images(project_id, split);

CREATE INDEX IF NOT EXISTS idx_dataset_images_project_augmented
ON dataset_images(project_id, is_augmented);

CREATE TABLE IF NOT EXISTS dataset_image_labels (
    project_id TEXT NOT NULL,
    image_id TEXT NOT NULL,
    class_id INTEGER NOT NULL,
    label_index INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (project_id, image_id, class_id, label_index),
    FOREIGN KEY (project_id, image_id) REFERENCES dataset_images(project_id, image_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_dataset_labels_project_class
ON dataset_image_labels(project_id, class_id);

CREATE TABLE IF NOT EXISTS dataset_image_boxes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL,
    image_id TEXT NOT NULL,
    box_index INTEGER NOT NULL,
    class_id INTEGER NOT NULL,
    x REAL NOT NULL,
    y REAL NOT NULL,
    w REAL NOT NULL,
    h REAL NOT NULL,
    UNIQUE (project_id, image_id, box_index),
    FOREIGN KEY (project_id, image_id) REFERENCES dataset_images(project_id, image_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_dataset_boxes_project_image
ON dataset_image_boxes(project_id, image_id);

CREATE TABLE IF NOT EXISTS runs (
    project_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    name TEXT,
    variant_id TEXT,
    status TEXT NOT NULL,
    config_json TEXT NOT NULL DEFAULT '{}',
    results_json TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT,
    completed_at TEXT,
    failed_at TEXT,
    PRIMARY KEY (project_id, run_id),
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_runs_project_created_at
ON runs(project_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_runs_project_status
ON runs(project_id, status);

CREATE TABLE IF NOT EXISTS run_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    metric_index INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT,
    UNIQUE (project_id, run_id, metric_index),
    FOREIGN KEY (project_id, run_id) REFERENCES runs(project_id, run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_run_metrics_project_run
ON run_metrics(project_id, run_id, metric_index);

CREATE TABLE IF NOT EXISTS variants (
    project_id TEXT NOT NULL,
    variant_id TEXT NOT NULL,
    run_id TEXT,
    name TEXT NOT NULL,
    base_model TEXT NOT NULL,
    hyperparameters_json TEXT NOT NULL DEFAULT '{}',
    metrics_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT,
    PRIMARY KEY (project_id, variant_id),
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_variants_project_created_at
ON variants(project_id, created_at DESC);

CREATE TABLE IF NOT EXISTS deployments (
    project_id TEXT NOT NULL,
    variant_id TEXT NOT NULL,
    deployment_id TEXT NOT NULL,
    run_id TEXT,
    target_id TEXT NOT NULL,
    target_label TEXT,
    format TEXT NOT NULL,
    package_dir TEXT,
    package_zip_path TEXT,
    artifact_path TEXT,
    readme_path TEXT,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (project_id, variant_id, deployment_id),
    FOREIGN KEY (project_id, variant_id) REFERENCES variants(project_id, variant_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_deployments_lookup_id
ON deployments(deployment_id);

CREATE INDEX IF NOT EXISTS idx_deployments_project_variant_created_at
ON deployments(project_id, variant_id, created_at DESC);

CREATE TABLE IF NOT EXISTS inference_images (
    project_id TEXT NOT NULL,
    image_id TEXT NOT NULL,
    image_file TEXT NOT NULL,
    original_name TEXT,
    created_at TEXT NOT NULL,
    last_tested_at TEXT,
    PRIMARY KEY (project_id, image_id)
);

CREATE INDEX IF NOT EXISTS idx_inference_images_project_last_tested
ON inference_images(project_id, last_tested_at DESC);

CREATE TABLE IF NOT EXISTS inference_variant_results (
    project_id TEXT NOT NULL,
    image_id TEXT NOT NULL,
    variant_id TEXT NOT NULL,
    tested_at TEXT NOT NULL,
    task TEXT,
    predictions_json TEXT NOT NULL DEFAULT '[]',
    latency_ms REAL,
    fps_est REAL,
    feedback TEXT,
    feedback_origin TEXT,
    PRIMARY KEY (project_id, image_id, variant_id),
    FOREIGN KEY (project_id, image_id) REFERENCES inference_images(project_id, image_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_inference_variant_results_project_variant
ON inference_variant_results(project_id, variant_id, tested_at DESC);

CREATE TABLE IF NOT EXISTS autolabel_suggestions (
    project_id TEXT NOT NULL,
    variant_id TEXT NOT NULL,
    image_id TEXT NOT NULL,
    run_id TEXT,
    predicted_class_id INTEGER,
    predicted_class_name TEXT,
    confidence REAL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    accepted_at TEXT,
    PRIMARY KEY (project_id, variant_id, image_id),
    FOREIGN KEY (project_id, image_id) REFERENCES dataset_images(project_id, image_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_autolabel_variant_status
ON autolabel_suggestions(project_id, variant_id, status);

CREATE TABLE IF NOT EXISTS dataset_versions (
    project_id TEXT NOT NULL,
    version_id TEXT NOT NULL,
    name TEXT NOT NULL,
    task TEXT NOT NULL,
    source_count INTEGER NOT NULL DEFAULT 0,
    selected_count INTEGER NOT NULL DEFAULT 0,
    rules_json TEXT NOT NULL DEFAULT '{}',
    class_counts_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (project_id, version_id),
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_dataset_versions_project_created_at
ON dataset_versions(project_id, created_at DESC);

CREATE TABLE IF NOT EXISTS dataset_version_images (
    project_id TEXT NOT NULL,
    version_id TEXT NOT NULL,
    image_id TEXT NOT NULL,
    image_index INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (project_id, version_id, image_id),
    FOREIGN KEY (project_id, version_id) REFERENCES dataset_versions(project_id, version_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_dataset_version_images_project_version_order
ON dataset_version_images(project_id, version_id, image_index);

CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS endpoint_cache (
    cache_key TEXT PRIMARY KEY,
    signature TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_endpoint_cache_updated_at
ON endpoint_cache(updated_at);
