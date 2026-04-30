# SoulX-FlashTalk Ascend 适配补丁

**路径约定**：下文若未特别说明，当前工作目录均为 **OmniRT 仓库根目录**（包含 `model_backends/`、`.omnirt/`）。补丁文件路径均相对该根目录书写。

本目录包含 **OmniRT 维护的、相对上游 GitHub `SoulX-FlashTalk` 的 Ascend 910B 运行补丁**，用于在 **不依赖 `xformers`**、以 **`torch_npu` + HCCL** 为主的环境下跑通 FlashTalk 推理与 OmniRT WebSocket 服务。

## 补丁文件

| 文件 | 说明 |
|------|------|
| `soulx-flashtalk-ascend-omnirt.patch` | 统一 diff，仅作用于 SoulX 仓库内的 `flash_talk/` 树。 |

**生成基线（供对照上游是否漂移）**：补丁相对官方仓库 **`origin/main` @ `f54dc79511e001472a4e46e14179d0db981c1caf`** 生成。若上游大幅改版，`git apply` 可能出现冲突，需要在新的上游 commit 上重新生成补丁（见文末）。

## 谁需要打补丁

- 从 **GitHub 裸克隆** `SoulX-AILab/SoulX-FlashTalk` 且未做任何本地 Ascend 适配时，常见问题是：`multitalk_attention` 等路径强依赖 **`xformers`**，而 Ascend 侧 venv 往往不安装该包。
- 打补丁后，会引入 **`flash_talk/src/attention_ops.py`**、**`accelerator.py`** 等，并在注意力与并行路径上走 NPU 友好实现；同时包含 **`infer_params.yaml` 的默认实时参数**（如 29 帧 / 1 motion overlap 等），可按业务再改。

## 用法

默认 SoulX 由 OmniRT 装在 **`.omnirt/model-repos/SoulX-FlashTalk`**。在 OmniRT 仓库根目录执行：

```bash
# 仅检查能否应用，不修改工作区
git -C .omnirt/model-repos/SoulX-FlashTalk apply --check model_backends/flashtalk/patches/soulx-flashtalk-ascend-omnirt.patch

# 真正应用
git -C .omnirt/model-repos/SoulX-FlashTalk apply model_backends/flashtalk/patches/soulx-flashtalk-ascend-omnirt.patch
```

或使用本目录脚本（参数为 SoulX 仓库根目录，须含 `flash_talk/`；默认示例如下）：

```bash
bash model_backends/flashtalk/patches/apply_soulx_flashtalk_ascend_patch.sh .omnirt/model-repos/SoulX-FlashTalk
```

撤销（在未提交前、且补丁未与其它手工修改混杂时），仍在 OmniRT 根目录：

```bash
git -C .omnirt/model-repos/SoulX-FlashTalk apply --reverse model_backends/flashtalk/patches/soulx-flashtalk-ascend-omnirt.patch
```

若提示补丁已应用，可先 `git apply --reverse --check` 判断是否已打过。

## 与 `omnirt runtime install` 的关系

`omnirt runtime install flashtalk --device ascend` 在 **成功克隆或更新 SoulX 仓库且存在 `.git`** 时，会在安装流程中 **自动尝试** 应用上述补丁：

- `git apply --check` 通过 → 执行 `git apply`。
- `git apply --reverse --check` 通过 → 视为 **已打过补丁**，跳过。
- 两者均失败 → **安装失败并给出错误**，避免在未知状态下继续装 venv。

若你使用 **`--repo-dir` 指向自己的 fork** 且已包含等价改动，补丁可能冲突；此时应使用已适配的 fork，或暂时从 OmniRT 仓库中移走/重命名补丁文件以跳过自动应用（不推荐长期使用）。

## 如何重新生成补丁

在「已验证可跑」的 SoulX 工作区上（相对干净的上游 + 你的 Ascend 改动）。SoulX 根目录仍用默认 **`.omnirt/model-repos/SoulX-FlashTalk`** 时，在 **OmniRT 仓库根目录** 执行：

```bash
git -C .omnirt/model-repos/SoulX-FlashTalk fetch origin
# 确认基线 commit，例如 origin/main
git -C .omnirt/model-repos/SoulX-FlashTalk add flash_talk/
git -C .omnirt/model-repos/SoulX-FlashTalk diff --cached origin/main -- flash_talk/ > model_backends/flashtalk/patches/soulx-flashtalk-ascend-omnirt.patch
git -C .omnirt/model-repos/SoulX-FlashTalk restore --staged flash_talk/
```

提交 OmniRT PR 前请在 **全新 shallow clone** 的 SoulX 上执行 `git apply --check` 做回归。
