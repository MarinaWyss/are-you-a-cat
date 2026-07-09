# Contributing

Thanks for your interest in improving **Are You A Cat?**

## Getting set up

```bash
python -m venv venv
source venv/bin/activate
make install
```

## Before you open a PR

1. **Lint**: `make lint` (flake8 must pass — CI enforces it).
2. **Test**: `make test` (add tests for any new behavior).
3. Keep changes focused and describe them in the PR body.

## Conventions

- Load configuration through `steps.config_loader.load_config` — never open
  `config.yaml` directly.
- Get loggers via `steps.logging_config.get_logger(__name__)` rather than
  calling `logging.basicConfig` in individual modules.
- The CNN topology lives in `model/architecture.py`. If you change the network,
  change it there so training and tuning stay in sync.
- Preprocess images through `steps.preprocessing` so training and serving apply
  identical transforms.

## Updating the changelog

Add a bullet under the `Unreleased` section of `CHANGELOG.md` describing your
change.
