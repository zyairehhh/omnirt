# `omnirt.requests`

Typed request builders — one keyword-only constructor per task surface. The auto-rendered signatures and docstrings live on the default-locale page: [→ API Reference / omnirt.requests](../../api_reference/requests/).

Exported builders:

- **`text2image(*, model, prompt, negative_prompt=None, backend="auto", adapters=None, **config) -> TextToImageRequest`**
- **`text2video(*, model, prompt, negative_prompt=None, num_frames=None, fps=None, backend="auto", adapters=None, **config) -> TextToVideoRequest`**
- **`image2image(*, model, image, prompt, negative_prompt=None, backend="auto", adapters=None, **config) -> ImageToImageRequest`**
- **`image2video(*, model, image, prompt=None, negative_prompt=None, num_frames=None, fps=None, backend="auto", adapters=None, **config) -> ImageToVideoRequest`**
- **`audio2video(*, model, image, audio, prompt=None, backend="auto", adapters=None, **config) -> AudioToVideoRequest`**
- **`inpaint(*, model, image, mask, prompt, negative_prompt=None, backend="auto", adapters=None, **config) -> InpaintRequest`**
- **`edit(*, model, image, prompt, backend="auto", adapters=None, **config) -> EditRequest`**

All builders accept arbitrary `**config` keyword arguments that flow into `GenerateRequest.config` (including `preset`, `width`, `height`, `num_inference_steps`, `guidance_scale`, `seed`, etc.).
