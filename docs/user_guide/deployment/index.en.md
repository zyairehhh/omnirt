# Deployment

OmniRT's deployment surface splits along two axes: **hardware backend** and **network environment**.

| Scenario | Start here |
|---|---|
| NVIDIA GPU production | [CUDA deployment](cuda.md) |
| Ascend Atlas / 910 / 910B | [Ascend backend](ascend.md) |
| Domestic / intranet / offline | [Domestic deployment](china_mirrors.md) |
| Containerized (Docker / k8s) | [Docker & containers](docker.md) |
| Gateway + workers + Redis / OTLP | [Distributed serving](distributed_serving.md) |

!!! tip "Validate before deploying"
    Before touching real hardware, run `omnirt validate` and `omnirt generate --dry-run` against `--backend cpu-stub` to confirm your request contract and model registry. See [Quickstart](../../getting_started/quickstart.md).

If your target deployment needs async jobs, cross-process job state, Prometheus scraping, or remote workers, continue with [Distributed serving](distributed_serving.md).

## Documentation Versions

The published site uses versioned docs: `/latest/` points at the current main docs, and formal releases are kept under paths such as `/vX.Y.Z/`. For production deployments, prefer the docs version that matches the Python package or Docker image tag you installed.
