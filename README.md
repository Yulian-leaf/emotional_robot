# 多模态情感识别系统

## 结构说明
- backend/ FastAPI 后端，负责视频上传、模态推理、融合、API
- frontend/ React 前端，负责上传、播放、时间轴可视化
- scripts/ ffmpeg 脚本
- docker-compose.yml 一键部署

## 快速开始
1. 安装 Docker
2. 在项目根目录运行 `docker-compose up --build`
3. 访问 http://localhost:3000 前端，上传视频体验

## 本地开发（Windows / 非 Docker）

如果你在 VS Code 里遇到 `npm` 识别不了、或 `Port 3000 is already in use`：

- 已在 `.vscode/settings.json` 里为 VS Code 集成终端预置了 Node/NPM 的 `Path`（重开一个终端标签页生效）
- 也可以直接用脚本一键启动前端（会自动停止占用 3000 的进程）：

```powershell
PowerShell -ExecutionPolicy Bypass -File .\scripts\start_frontend.ps1
```

## 模型配置（推荐）

后端支持通过环境变量或 warmup 接口指定模型（Hugging Face 模型 ID 或本地目录）。

### 关键环境变量
- `TEXT_ASR_MODEL`：ASR 转写模型（Whisper）。不设置则视频的 `transcript` 可能为空。
- `TEXT_MODEL_ID`：文本情感模型（用于对转写文本做情感分类）。
- `AUDIO_MODEL_ID`：语音情感模型（SER）。
- `IMAGE_MODEL_ID`：人脸/表情情感模型（FER）。

### ASR（Whisper）模型选择
转写速度与准确率一般是“模型越大越准但越慢”。常见可选：
- `openai/whisper-tiny`：最快，适合先跑通链路。
- `openai/whisper-base` / `openai/whisper-small`：更准，但 CPU 上会明显更慢。

### 用 warmup 一次性设置（Windows PowerShell）
用 `Invoke-RestMethod` 发送 JSON（比 `curl.exe -d` 更不容易踩转义坑）：

```powershell
$payload = '{
	"text_asr_model":"openai/whisper-base",
	"text_model_id":"D:/python-learn/program/models/text-emotion",
	"audio_model_id":"D:/python-learn/program/models/audio-ser",
	"image_model_id":"D:/python-learn/program/models/image-fer",
	"text_debug":1,
	"audio_debug":1,
	"image_debug":1,
	"image_force_hf":1
}'

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/models/warmup -ContentType 'application/json' -Body $payload
```

说明：如果你机器跑 `whisper-base/small` 太慢，把 `text_asr_model` 改回 `openai/whisper-tiny` 即可。

## 模型接入与微调
- 各模态服务已预留接口，支持 Hugging Face、PyTorch 等模型集成
- 可扩展微调 API，详见 backend/services/

## 目录结构
- backend/app.py 主入口
- backend/services/ 各模态与融合服务
- frontend/src/ 前端主页面与组件
- scripts/ 视频处理脚本
