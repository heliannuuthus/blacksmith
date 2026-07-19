# RasterRecast

RasterRecast 是一个本机纯 AI 图片文字编辑工具。用户在 Qt 6 桌面界面中框选原文字、输入
新的汉字或数字，由固定的 `FLUX-Text Multi-Resolution` 引擎完成字形约束、字体风格迁移、
背景修复和局部重绘。

图片仅通过当前用户可访问的 Unix Socket 传给项目自己的本机 AI 进程，不会监听网络端口，
也不会上传到第三方推理服务。项目不包含 OCR、字体匹配、LaMa 擦除或 FreeType 文字贴图
等传统替换路径。

## 架构

- 桌面端：PySide6，负责文件选择、框选、撤销、预览和导出。
- 内置推理运行时：独立 Python 进程，直接加载 Diffusers、MMGP 和 FLUX-Text。
- 基础模型：FLUX.1 Fill Dev 的 MMGP 量化版本。
- 文字能力：[AMAP-ML/FluxText](https://github.com/AMAP-ML/FluxText) Multi-Resolution LoRA。
- 3080 10GB 默认使用 `LowRAM_LowVRAM` 分层卸载和固定 28 步推理。

FLUX-Text 会将新文字生成目标字形控制图，但控制图只作为扩散模型的条件，不会直接贴到
原图。最终字形、原字体风格、光影、材质和背景均由模型生成。

运行时不导入、不启动也不连接 ComfyUI，不使用节点工作流、HTTP 服务或 `8188` 端口。

## 首次安装模型运行时

运行时使用隔离环境，目录结构如下：

```text
models/fluxtext-worker/
├── .venv/
├── fluxtext-runtime/        # FLUX-Text 推理源码
└── models/
    ├── AIFSH/flux1-fill-dev-mmgp/
    ├── diffusion_models/
    ├── text_encoders/
    └── loras/
```

当前机器已经安装完成，并可直接复用原来下载的权重，无须重新下载。新环境使用
`worker-requirements.txt` 安装独立运行时依赖。这里固定 OpenCV 4.10；5.x 改变了
`minAreaRect` 的角度表示，会把横排字形条件误转 90°。

模型文件总计约 19.4GB：

| 文件 | 放置目录 |
|---|---|
| `flux1_fill_dev_transformer_mmgp.safetensors` | `models/diffusion_models/` |
| `flux1_fill_dev_text_encoder_2_mmgp.safetensors` | `models/text_encoders/` |
| FLUX-Text Multi-Resolution LoRA | `models/loras/flux-text-multisize.safetensors` |

前两个文件来自
[`tt98/flux1-fill-dev-mmgp`](https://huggingface.co/tt98/flux1-fill-dev-mmgp)，LoRA 来自
[`GD-ML/FLUX-Text`](https://huggingface.co/GD-ML/FLUX-Text)。此外需要约 580MB 的
tokenizer、CLIP 和 VAE 基础组件。

安装目录位于 `blacksmith/models/fluxtext-worker` 时，RasterRecast 会自动发现。其他目录可通过
`RASTER_RECAST_WORKER_DIR=/path/to/fluxtext-worker` 指定；仍然只需启动桌面端。

三个模型文件名也可以分别通过 `RASTER_RECAST_FLUX_TRANSFORMER`、
`RASTER_RECAST_FLUX_TEXT_ENCODER`、`RASTER_RECAST_FLUX_TEXT_LORA` 覆盖。

## 启动桌面端

在 `blacksmith` 根目录执行：

```bash
uv run raster-recast
```

这一个命令会打开 RasterRecast，并在后台自动拉起本机 AI 引擎；关闭 GUI 时，由 GUI 启动的
后台进程也会自动退出。

也可以启动时直接打开图片：

```bash
uv run raster-recast input.png
```

## 使用流程

1. 选择或拖入本机图片。
2. 在预览画布上框住需要替换的原文字。
3. 输入目标汉字或数字。
4. 点击“开始 AI 替换”。
5. 预览、撤销或继续框选，最后选择目录导出。

“参考上下文”控制模型在选区外额外观察的像素范围。默认 96px；背景、透视或光影复杂时
可以提高。模型生成带上下文的局部图像，桌面端只把选区范围合回原图，避免框外漂移。

## 验证

```bash
uv sync --extra dev
uv run --extra dev ruff check apps/raster-recast
uv run --extra dev mypy apps/raster-recast/src
uv run --extra dev pytest apps/raster-recast/tests
```
