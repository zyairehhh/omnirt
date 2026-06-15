# v0.1.0 Release Verification

This runbook records the verification evidence for the OmniRT v0.1.0 release. Use it to confirm that the GitHub Release, Python artifacts, and GHCR Docker image were produced from the intended release commit.

## Release Coordinates

| Item | Value |
| --- | --- |
| Release branch | `release-v0.1.0` |
| Release commit | `5e34b6fa469f86c354da7dd047320a0a0829a5f6` |
| Release tag | `v0.1.0` |
| GitHub Release | <https://github.com/datascale-ai/omnirt/releases/tag/v0.1.0> |
| Release workflow | <https://github.com/datascale-ai/omnirt/actions/runs/27564580582> |

## GitHub Release Artifacts

The release workflow completed successfully and attached these Python artifacts to the GitHub Release:

- `omnirt-0.1.0-py3-none-any.whl`
- `omnirt-0.1.0.tar.gz`

Verification command:

```bash
python -m build
python -m twine check dist/*
```

Observed result on the release worktree:

```text
Checking dist/omnirt-0.1.0-py3-none-any.whl: PASSED
Checking dist/omnirt-0.1.0.tar.gz: PASSED
```

## Docker Image

OmniRT publishes one server image for the v0.1.0 release:

```text
ghcr.io/datascale-ai/omnirt-server:v0.1.0
```

After the GHCR package was made public, anonymous manifest inspection succeeded from the `146` host without Docker login credentials:

```bash
docker manifest inspect ghcr.io/datascale-ai/omnirt-server:v0.1.0
```

Observed manifest digest hint:

```text
omnirt-server sha256:2ae2fafcdb6eca44d213aba35916f6cb10a299ee3cdfd59998a5f15b667fa04d
```

The image was also built and smoke-tested locally on the `146` release worktree. A full anonymous pull was started after the package became public, but the network retried on a large layer for several minutes; manifest access is the release gate for public GHCR availability, and the runtime smoke test below verifies the server image behavior.

## Runtime Smoke Test

The following smoke test was run on `8.92.9.146` from `/data2/zhongyi/omnirt-release-v0.1.0`.

```text
omnirt-server-v010-smoke Up 8 seconds 0.0.0.0:18081->8000/tcp, :::18081->8000/tcp
omnirt_health_ok {"ok": false, "adapter_schema": "text2audio.service.v1", "statuses": [{"id": "indextts", "connected": false, "reason": "runtime_disabled"}]}
```

The `runtime_disabled` status is expected for a lightweight server image without model weights or hardware-specific runtime setup.

## Acceptance Checklist

- Release branch exists on `datascale-ai/omnirt`.
- `v0.1.0` tag points to the release commit.
- GitHub Release exists and contains wheel and source distribution assets.
- Release workflow completed successfully.
- GHCR package visibility is public enough for anonymous manifest inspection.
