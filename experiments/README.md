# Experiment Registry

Each run appends one JSON line to `experiments/registry.jsonl`.

Required fields:

- `run_id`
- `timestamp_utc`
- `git_commit`
- `dataset_version`
- `seed`
- `model_name`
- `config`
- `metrics`

Recommended extras:

- `training_time_seconds`
- `compute`

The baseline script writes the registry entry automatically after generating a prediction file.

