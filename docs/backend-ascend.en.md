# Ascend Backend Notes

The Ascend backend follows the same contract as CUDA, but its compile path is intentionally conservative.

## Execution model

- backend name: `ascend`
- device name: `npu`
- compile attempt: `torch_npu.npu.graph_mode()` when available
- fallback behavior: if graph-mode setup fails or a module cannot be compiled, the runtime records the failure and keeps the eager module

## Current expectations

- requests can explicitly target `--backend ascend`
- the same `GenerateRequest` schema is used for CUDA and Ascend
- capability reporting exposes dtype options and compile availability
- backend fallback attempts are preserved in `RunReport.backend_timeline`

## Validation workflow

The repository ships Ascend smoke tests, but they run only when all of the following are true:

- `torch_npu` is installed
- Diffusers runtime dependencies are installed
- model sources are provided through `OMNIRT_SDXL_MODEL_SOURCE` and `OMNIRT_SVD_MODEL_SOURCE`
- the tests execute on an Ascend-capable host

Without those prerequisites the tests skip rather than failing noisily.
