# 文档发布

当前仓库使用 MkDocs 和 Material for MkDocs 构建静态文档站，并通过 GitHub Pages 发布。

## 本地预览

安装文档工具链：

```bash
python -m pip install -e '.[docs]'
```

启动本地预览服务：

```bash
mkdocs serve
```

按 CI 的方式构建站点：

```bash
mkdocs build --strict
```

`--strict` 是当前 PR 和 Pages workflow 的默认校验模式，因为它能尽早暴露断链和导航配置错误。

## GitHub Pages 发布

仓库已经提供独立 workflow：`.github/workflows/docs.yml`。

发布流程包含：

- 在 `main` 分支 push 时自动触发
- 也支持通过 `workflow_dispatch` 手动触发
- 用 `pip install -e '.[docs]'` 安装文档依赖
- 通过 `mkdocs build --strict` 构建当前 ref 的文档
- 用 `mike` 把当前文档发布到 `gh-pages` 分支里的版本目录
- 从 `gh-pages` 分支导出完整版本化站点到 `site/`
- 如果仓库已经启用 GitHub Pages，则上传 `site/` 目录作为 Pages artifact
- 如果仓库已经启用 GitHub Pages，则通过官方 GitHub Pages actions 流程部署

## 版本化文档

默认的 `main` push 会发布两个入口：

- `/main/`：当前 main 分支文档
- `/latest/`：指向当前 main 文档的别名

发布正式版本文档时，通过手动触发 docs workflow，把 `version` 设为具体版本号，例如 `v0.1.0`。如需把某个版本设为默认入口，把 `default_version` 设为对应版本或别名。`mike` 会维护 `versions.json`，Material for MkDocs 会在站点里显示版本切换器。

## 站点 URL 与路径模型

这个仓库使用的是 project site，而不是 user site。

- 仓库：`datascale-ai/omnirt`
- 对外地址：`https://datascale-ai.github.io/omnirt/`

这意味着文档发布后会挂在 `/omnirt/` 路径下，本地预览和站内链接都应按这个前提设计。

## 维护约定

- 保持 `README.md` 聚焦仓库首页摘要，不承载过长的使用手册
- 更长的用户文档优先放进 `docs/`
- 尽量保持文档路径稳定，主要通过 `mkdocs.yml` 调整导航
- 在合并较大的文档改动前，先跑一次 `mkdocs build --strict`

如果仓库还没有启用 GitHub Pages，workflow 仍会维护 `gh-pages` 里的版本化文档内容，但不会执行 Pages artifact 部署。需要在仓库设置里把 Pages source 切到 GitHub Actions；完成这一步后，后续线上发布由 workflow 接管。
