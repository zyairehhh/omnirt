# v0.1.0 发版验证

本文记录 OmniRT v0.1.0 的发版验证证据，供维护者确认 GitHub Release、Python artifacts 和 GHCR Docker 镜像是否来自预期的 release commit。

## 发版坐标

| 项目 | 值 |
| --- | --- |
| 发版分支 | `release-v0.1.0` |
| 发版 commit | `5e34b6fa469f86c354da7dd047320a0a0829a5f6` |
| 发版 tag | `v0.1.0` |
| GitHub Release | <https://github.com/datascale-ai/omnirt/releases/tag/v0.1.0> |
| Release workflow | <https://github.com/datascale-ai/omnirt/actions/runs/27564580582> |

## GitHub Release Artifacts

Release workflow 已成功完成，并在 GitHub Release 中挂载以下 Python artifacts：

- `omnirt-0.1.0-py3-none-any.whl`
- `omnirt-0.1.0.tar.gz`

验证命令：

```bash
python -m build
python -m twine check dist/*
```

release worktree 上观察到的结果：

```text
Checking dist/omnirt-0.1.0-py3-none-any.whl: PASSED
Checking dist/omnirt-0.1.0.tar.gz: PASSED
```

## Docker 镜像

OmniRT 在 v0.1.0 发布一个 server 镜像：

```text
ghcr.io/datascale-ai/omnirt-server:v0.1.0
```

GHCR package 改为 public 后，在没有 Docker 登录态的 `146` 验证机上匿名 manifest 检查成功：

```bash
docker manifest inspect ghcr.io/datascale-ai/omnirt-server:v0.1.0
```

观察到的 manifest digest hint：

```text
omnirt-server sha256:2ae2fafcdb6eca44d213aba35916f6cb10a299ee3cdfd59998a5f15b667fa04d
```

该镜像也已经在 `146` release worktree 上完成本地 build 和 smoke test。package 改为 public 后曾启动完整匿名 pull，但大层下载持续重试数分钟；发版验收以 manifest 可匿名读取作为 GHCR public 可用性门槛，下面的 runtime smoke test 验证 server 镜像行为。

## Runtime Smoke Test

以下 smoke test 在 `8.92.9.146` 上执行，release worktree 位于 `/data2/zhongyi/omnirt-release-v0.1.0`。

```text
omnirt-server-v010-smoke Up 8 seconds 0.0.0.0:18081->8000/tcp, :::18081->8000/tcp
omnirt_health_ok {"ok": false, "adapter_schema": "text2audio.service.v1", "statuses": [{"id": "indextts", "connected": false, "reason": "runtime_disabled"}]}
```

`runtime_disabled` 对轻量 server 镜像是预期状态，因为镜像没有内置模型权重或硬件相关 runtime。

## 验收清单

- `datascale-ai/omnirt` 上存在 release 分支。
- `v0.1.0` tag 指向 release commit。
- GitHub Release 已创建，并包含 wheel 与 source distribution assets。
- Release workflow 成功完成。
- GHCR package visibility 已足够公开，支持匿名 manifest inspection。
