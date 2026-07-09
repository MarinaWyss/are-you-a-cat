# Changelog

All notable changes to this project are documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.2.0] - Unreleased

### Added
- `steps/config_loader.py`: a single cached, validated entry point for loading
  `config.yaml`, replacing the inline `open('steps/config.yaml')` calls that
  were scattered across the code base.
- `steps/logging_config.py`: project-wide, consistently formatted logging.
- `steps/seed.py`: reproducible seeding of Python, NumPy, and TensorFlow
  (resolves the long-standing `# TODO random seed`).
- `steps/preprocessing.py`: image preprocessing shared by the training and
  serving paths so they can no longer diverge.
- `model/architecture.py`: the CNN is now defined once and reused by both the
  fixed-config training path and the Keras Tuner search (resolves the
  copy-paste `# TODO` in `hyperparameter_tuning.py`).
- `model/predictor.py`: maps a raw probability to a structured, testable
  verdict.
- Input validation in the Streamlit app (file size + decodability).
- New tests: config loader, evaluator, predictor, preprocessing, and model
  architecture.
- Developer tooling: `Makefile`, `pyproject.toml`, `.flake8`, root `conftest.py`,
  and a `lint` job in CI.

### Changed
- `model/evaluator.py`: collapsed three near-identical try/except metric methods
  into one shared helper.
- `steps/utils.py`: uses the shared preprocessing helper and a modern Pillow
  resampling filter (no more deprecated `Image.ANTIALIAS`).
- Pipeline steps and run scripts now load config and logging through the new
  shared modules.

### Fixed
- `requirements.txt`: pinned/added `keras-tuner`, `streamlit`, and `flake8`,
  which were used but previously undeclared.
