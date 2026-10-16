# Qwen-Image-2.1 本地运行

统一文生图 / 图像编辑 / RGBA 透明图生成（7B DiT + 文本编码器 + VAE）。
本目录脚本针对 **24GB 显存**（RTX 4500 Ada / 4090 等）优化。

> 模型为 **Qwen Research License**，仅限研究非商用，商用需单独授权。

## 环境准备

推荐 Python 3.11 ~ 3.12（3.13 需 torch>=2.5，通常也可用）。一键脚本会建虚拟环境、装依赖，并把模型缓存重定向到本目录：

```bash
cd F:\Code\qwen-image-2.1
setup.bat
```

手工等价命令：

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
pip install --index-url https://download.pytorch.org/whl/cu130 torch torchvision
pip install -r requirements.txt
```

> **重要**：Windows 上直接 `pip install torch` 会装成 **CPU 版**（`cuda_available=False`），必须用上面的 PyTorch cu 源。本机驱动 CUDA 13.0，故用 `cu130`（若为其他版本改用 cu128/cu126 等）。
> 验证：`python -c "import torch; print(torch.cuda.is_available())"` 应输出 `True`。

## 模型下载位置

脚本通过 `HF_HOME` 把权重放到项目目录下，**不占 C 盘**：

```
F:\Code\qwen-image-2.1\.hf\hub\models--Qwen--Qwen-Image-2.1
```

约 30GB+。想删除直接删 `.hf` 文件夹即可。

## 用法

用 `run.bat` 运行会自动设置 `HF_HOME` / `HF_ENDPOINT`（国内镜像）+ 激活 venv：

```bash
run.bat --prompt "A neon shop sign that reads \"QWEN IMAGE 2.1\", rainy night, reflections on wet pavement"
```

> 也可手动激活环境后直接 `python generate.py ...`，但需自行 `set HF_HOME=F:\Code\qwen-image-2.1\.hf` 与 `set HF_ENDPOINT=https://hf-mirror.com`。

### 文生图

指定比例 / 步数 / 输出：

```bash
run.bat --prompt "一只戴宇航头盔的柴犬" --ratio 16:9 --steps 40 --output dog.png
```

### 图像编辑（以图生图）

传 `--image` 输入图 + `--prompt` 描述改动。不指定尺寸时，输出**自动跟随输入图的宽高比**（边长由 `--resolution` 决定，默认 2048）：

```bash
run.bat --image input.png --prompt "把背景换成日落海滩"
```

换风格 / 局部改动都靠提示词，例如：

```bash
run.bat --image photo.jpg --prompt "turn it into a watercolor painting" --output wc.png
```

多图参考（最多 10 张，列表作为一组参考）：

```bash
run.bat --image a.png --image b.png --prompt "把 <image1> 的主体放进 <image2> 的场景"
```

> 说明：`QwenImage21Pipeline` 的 `image` 接受 PIL / numpy，可以是单张或列表。
> 它**没有独立 mask 参数**，局部编辑（圈选、涂鸦、遮罩）通过对输入图绘制标注、或在提示词里描述区域来实现。
> `--resolution` 同时决定条件图的缩放边长；想更高清编辑相册照片可设 `--resolution 2048`（默认）。
> 默认 `--cfg 1.0`（模型建议不加引导）；若要用引导需同时给 `--negative-prompt` 并设 `--cfg 3.0` 之类。

#### 小尺寸预览

`--width` / `--height` 控制最终输出尺寸，`--resolution` 仍控制参考图缩放。可以生成 128x128 的小图：

```powershell
.\run.bat --image "input.jpg" --prompt "<image1> 的角色侧坐在花田里，双手抱膝，头侧向摄像机，微笑" --width 128 --height 128 --resolution 128 --count 10 --output-dir result
```

脚本关闭视觉处理器的二次缩放，让视觉编码器和 VAE 使用 pipeline 已统一缩放的参考图，避免 128 / 256 等小分辨率下的 `shape mismatch`。小图适合快速预览，但会丢失参考图和输出中的细节；需要更好的人物一致性时，可以保留 128x128 输出，将 `--resolution` 提高到 512 或 1024。

#### 编辑常用参数

| 参数 | 说明 |
|---|---|
| `--resolution` | 输出/条件图边长，默认 2048；编辑跟随输入宽高比 |
| `--cfg` | `true_cfg_scale`，默认 1.0；>1 需配 `--negative-prompt` |
| `--negative-prompt` | 负向提示词 |

### 连续生成多张候选图

两张参考图生成 10 张 1024x1024 的候选图，统一保存到 `candidates` 文件夹。在 PowerShell 中运行：

```powershell
.\run.bat --image "a.png" --image "b.png" --prompt "把 <image1> 的主体放进 <image2> 的场景，光照自然一致" --width 1024 --height 1024 --resolution 1024 --count 10 --output-dir candidates
```

- 模型和参考图只加载一次，逐张推理并保存，不同时生成 10 张以免显存不足。
- 默认种子从 42 递增到 51，文件名为 `output_01_seed42.png` 到 `output_10_seed51.png`。
- 可以用 `--seed` 指定起始种子，用 `--output pick.png` 将文件名前缀改为 `pick`。
- 不指定 `--output-dir` 时，保存在 `--output` 的父目录；指定后使用该目录和 `--output` 的文件名。
- 默认 `--count 1`，保留原来的单张文件名；重复运行相同命令会覆盖同名结果。
- 选中一张后，可以用文件名中的种子配合相同输入和参数、`--count 1` 单独生成。

### 透明图（RGBA）

```bash
run.bat --prompt "一只可爱的卡通恐龙贴纸" --transparent --output sticker.png
```

## 显存策略（--offload）

| 取值 | 显存峰值 | 速度 | 适用 |
|---|---|---|---|
| `model`（默认） | ~15–18GB | 较快 | **24GB 推荐** |
| `sequential` | <10GB | 很慢 | 更小显存 |
| `none` | >24GB | 最快 | 会 OOM，勿用 |

- 开了 `model`/`sequential` offload 后脚本不会再 `.to("cuda")`。
- offload 依赖 CPU 内存暂存权重，建议系统内存 **32GB 起**。
- `--dtype fp16` 可再省一点显存；`--cpu` 仅用于调试，极慢。

## 参数速查

| 参数 | 说明 |
|---|---|
| `--prompt` | 提示词（必填） |
| `--image` | 编辑输入图，可重复 |
| `--ratio` | `1:1 / 4:3 / 3:4 / 3:2 / 2:3 / 16:9 / 9:16` |
| `--width --height` | 自定义尺寸，优先于 `--ratio` |
| `--resolution` | 输出/条件图边长，默认 2048（编辑跟随输入宽高比） |
| `--steps` | 推理步数，默认 40 |
| `--cfg` | `true_cfg_scale`，默认 1.0（>1 需配负向提示词） |
| `--negative-prompt` | 负向提示词 |
| `--seed` | 起始随机种子，默认 42，多张生成时逐张递增 |
| `--count` | 连续生成张数，默认 1，必须大于 0 |
| `--output` | 输出路径，默认 output.png；多张时附加序号和种子 |
| `--output-dir` | 统一输出目录，自动创建；使用 `--output` 的文件名 |
| `--transparent` | 透明图模式 |
| `--offload` | `model / sequential / none` |
| `--model` | 换 HF id 或本地权重目录 |

## 支持比例与分辨率

```
1:1  2048x2048      4:3  2400x1792      3:4  1792x2400
3:2  2528x1696      2:3  1696x2528
16:9 2752x1536      9:16 1536x2752
```
