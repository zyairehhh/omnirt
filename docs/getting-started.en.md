# Getting Started

This guide is the shortest path from a fresh checkout to a validated OmniRT request and a local docs preview.

## Install

For general development:

```bash
python -m pip install -e '.[dev]'
```

To run real model pipelines, install runtime dependencies too:

```bash
python -m pip install -e '.[runtime,dev]'
```

To work on the documentation site:

```bash
python -m pip install -e '.[docs]'
```

If you want one environment for code, runtime, and docs:

```bash
python -m pip install -e '.[runtime,dev,docs]'
```

## Inspect the CLI

```bash
python -m omnirt --help
omnirt models
omnirt models flux2.dev
```

`omnirt models` is the quickest way to inspect the live registry without reading source code.

## Validate the first request

Start with a dry validation pass before using accelerator hardware:

```bash
omnirt validate \
  --task text2image \
  --model qwen-image \
  --prompt "a poster with a bold title" \
  --backend cpu-stub
```

You can validate a config file too:

```bash
omnirt validate --config request.yaml --json
```

## Run the first generation

```bash
omnirt generate \
  --task text2image \
  --model sd15 \
  --prompt "a lighthouse in fog" \
  --backend cuda \
  --preset fast
```

Example YAML request:

```yaml
task: text2image
model: flux2.dev
backend: auto
inputs:
  prompt: "a cinematic sci-fi city at sunrise"
config:
  preset: balanced
  width: 1024
  height: 1024
```

Run it with:

```bash
omnirt generate --config request.yaml --json
```

## Run tests

Fast local coverage:

```bash
pytest tests/unit tests/parity
```

Error-path integration coverage:

```bash
pytest tests/integration/test_error_paths.py
```

Hardware-backed CUDA and Ascend smoke tests are available in CI and skip locally unless the expected runtime packages and model directories are present.

## Preview the docs site

```bash
mkdocs serve
```

Build the static site with strict link checking:

```bash
mkdocs build --strict
```

The GitHub Pages deployment guide lives in [Publishing Docs](publishing-docs.md).
