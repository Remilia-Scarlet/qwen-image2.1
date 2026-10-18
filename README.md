# Qwen-Image-2.1 本地运行

统一文生图 / 图像编辑 / RGBA 透明图生成，基于官方 **Qwen/Qwen-Image-2.1-Turbo** 检查点（**8 步出图**）。

- **16GB 显存**：配合 Q8 量化 DiT（`--gguf`，DiT 常驻显存），单张 1024 约 **25s**。
- **24GB 显存**：无需量化，直接运行。

> 模型为 **Qwen Research License**，仅限研究非商用，商用需单独授权。

## 环境准备

推荐 Python 3.11 ~ 3.12。一键脚本会建虚拟环境、装依赖，并把模型缓存重定向到本目录（不占 C 盘）：

```bash
cd 项目目录
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

> **重要**：Windows 上直接 `pip install torch` 会装成 **CPU 版**（`cuda_available=False`），必须用上面的 PyTorch cu 源。示例驱动为 CUDA 13.0，故用 `cu130`（其他驱动版本相应改用 cu128/cu126 等）。
> 验证：`python -c "import torch; print(torch.cuda.is_available())"` 应输出 `True`。

## 模型下载

### Turbo 检查点（必装，约 30GB）

```powershell
download_turbo.bat
```

等价命令（HF 国内镜像 + 禁用 Xet，否则会绕开镜像报 401）：

```powershell
$env:HF_HOME="D:\qwen-image2.1\.hf"      # 改成你的项目目录
$env:HF_ENDPOINT="https://hf-mirror.com"
$env:HF_HUB_DISABLE_XET="1"
.venv\Scripts\hf.exe download Qwen/Qwen-Image-2.1-Turbo
```

> 下载中断（超时/断网）直接重新运行即可**断点续传**；完成后 `.hf` 里约 30GB。
> 8 步来自检查点自带的 `sample_sigmas`，脚本自动识别；`--steps` 无效，`--cfg` 保持默认 1.0 即可。

### Q8 量化 DiT（16GB 显存推荐，7.59GB）

```powershell
download_gguf.bat
```

## 用法

### 网页控制台

运行 `start_web.bat`，浏览器会打开 `http://127.0.0.1:7860`。控制台使用现有环境和本地 Turbo / Q8 权重，不自动下载或替换模型；`run.bat` 的命令行用法保持不变。

网页版的文本编码和 DiT 策略独立配置。显存常驻只针对组件的执行阶段，阶段结束后移回 CPU，为后续编码、迭代或 VAE 解码腾出显存。

可用 `start_web.bat --port 7861 --no-browser` 更换端口并关闭自动打开浏览器。`--model-dir <本地 Turbo 目录>` 和 `--gguf <本地 GGUF 文件>` 可指定其他权重位置；`--cpu` 仅用于较慢的 CPU 调试。

原始 PNG 保存在 `outputs/<任务编号>/`。接口保留当前会话最近 64 个已结束任务，重启只清除接口历史，不删除图片。停止按钮和 `Ctrl+C` 会等待当前操作到达安全停止点，保留已保存的结果。

用 `run.bat` 运行会自动设置 `HF_HOME` / `HF_ENDPOINT`（国内镜像）+ 激活 venv：

```bash
run.bat --prompt "A neon shop sign that reads \"QWEN IMAGE 2.1\", rainy night, reflections on wet pavement"
```

> 也可手动激活环境后直接 `python generate.py ...`，但需自行 `set HF_HOME=<项目目录>\.hf` 与 `set HF_ENDPOINT=https://hf-mirror.com`。

### 文生图

指定输出宽高和文件名；不指定尺寸时默认 2048 × 2048：

```bash
run.bat --prompt "一只戴宇航头盔的柴犬" --width 2752 --height 1536 --output dog.png
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

`--width` / `--height` 控制最终输出尺寸，`--resolution` 仍控制参考图缩放。可以生成 128x128 的小图快速预览：

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

两张参考图生成 10 张 1024x1024 的候选图，统一保存到 `candidates` 文件夹：

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

## 16GB 显存：Q8 量化 DiT（--gguf）

把 DiT 换成社区量化的 GGUF Q8_0（7.59GB，替代 13.25GB bf16），**整份常驻显存**，文本编码器/VAE 仍从官方 Turbo 加载：

```powershell
run.bat --image a.png --image b.png --prompt "把 <image1> 的主体放进 <image2> 的场景" `
        --width 1024 --height 1024 --resolution 1024 --output-dir batchtest --output q8.png `
        --gguf gguf\Qwen-Image-2.1-Turbo-Q8_0.gguf
```

16GB 显卡（RTX 5070 Ti）参考耗时（1024，双参考图，8 步）：

| 场景 | 耗时 |
|---|---|
| 单张（首次调用） | ~26s |
| 单张（同进程第二张起） | ~21s |
| 两张逐张 `--count 2` | ~46s |
| `--batch 2`（自动回退，见下） | ~47s |

- DiT 权重保持压缩格式（7.07GiB）常驻显存，Linear 前向时逐层现场反量化，无每步搬运；**传了 `--gguf` 就忽略 `--offload`**。
- VAE 解码前脚本会自动把 DiT 临时挪回 CPU（~1s），腾出解码峰值显存，下一张再挪回（~1.6s）；单张显存峰值 ~15.6GB。
  > 已接近 16GB 上限，运行前请关掉其他占显存的程序。
- `--batch > 1` 时两张图同时去噪的激活 + 常驻权重会超出 16GB，脚本会打印提示并自动回退为 DiT 逐层卸载（耗时同级，~47s/两张，任选方式）。
- 画质：Q8_0 量化接近无损（上游基准 LPIPS 0.037 / PSNR ~30dB），同种子出图与全精度有细微像素差异、构图一致；介意可自行对比。
- 回退：去掉 `--gguf` 即恢复官方 bf16（官方权重不受影响）。

## 24GB 显存

直接运行即可，默认 `--offload auto` 会自动选择 `model`（组件级卸载），速度较快。

## 显存策略（--offload）

| 取值 | 显存峰值 | 速度 | 适用 |
|---|---|---|---|
| `auto`（默认） | 视显存 | - | 自动：≥20GB→`model`，否则→`sequential` |
| `model` | ~15–18GB | 较快 | **24GB 推荐** |
| `sequential` | <10GB | 一般 | **16GB（不用 Q8 时）**；受 PCIe 带宽限制 |
| `none` | >24GB | 最快 | 会 OOM，勿用 |

> `--batch 2` 把两张图合成一次调用：文本编码只做一次、DiT 每步多算一张，每张均摊时间约降 1/3；显存估算：1024 下 N=2 约 4–6.5GB、N=4 约 8–12GB；2K/参考图@2048 时只建议 N=2。
> `model` 在 16GB 上必溢出：`text_encoder` 本身约 16GB，整包上卡直接放不下；16GB 请用上文 `--gguf`。
> offload 依赖 CPU 内存暂存权重，建议系统内存 **32GB 起**。

- 开了 `model`/`sequential` offload 后脚本不会再对管线整体 `.to("cuda")`。
- 每次生成输出 `[timing]` 分段耗时（`text_encoder` / `transformer` / `vae`），便于判断瓶颈。
- `--dtype`：`bf16`（默认）/ `fp16` / `fp32`。`fp16` 与 `bf16` 都是 2 字节/参数，**显存占用相同**，只是数值范围/精度差异；`--cpu` 仅用于调试，极慢。

## 参数速查

| 参数 | 说明 |
|---|---|
| `--prompt` | 提示词（必填） |
| `--image` | 编辑输入图，可重复 |
| `--width --height` | 同时指定输出宽高；省略时文生图默认 2048 × 2048，编辑模式自动跟随输入宽高比 |
| `--resolution` | 输出/条件图边长，默认 2048（编辑跟随输入宽高比） |
| `--steps` | 推理步数，默认 40（Turbo 固定 8 步，忽略此参数） |
| `--cfg` | `true_cfg_scale`，默认 1.0（>1 需配负向提示词） |
| `--negative-prompt` | 负向提示词 |
| `--seed` | 起始随机种子，默认 42，多张生成时逐张递增 |
| `--count` | 连续生成张数，默认 1，必须大于 0 |
| `--batch` | 每次管线调用出图数，默认 1；>1 时共享文本编码并按 batch 去噪（吃显存，见下） |
| `--output` | 输出路径，默认 output.png；多张时附加序号和种子 |
| `--output-dir` | 统一输出目录，自动创建；使用 `--output` 的文件名 |
| `--transparent` | 透明图模式 |
| `--offload` | `auto / model / sequential / none`（`--gguf` 时忽略） |
| `--dtype` | `bf16`（默认）/ `fp16` / `fp32` |
| `--model` | 换 HF id 或本地权重目录 |
| `--gguf` | 用 GGUF 量化文件替换 DiT（如 `gguf\Qwen-Image-2.1-Turbo-Q8_0.gguf`）；DiT 整份常驻显存，`--batch > 1` 时自动回退逐层卸载 |
