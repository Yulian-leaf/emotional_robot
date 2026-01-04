# emotion_robot 项目说明文档（用于绘制流程图）

> 目标：把本项目的**端到端链路**与**各子模块**讲清楚，让你能直接在 Visio / draw.io / Mermaid 中画出对应流程图。
>
> 本文基于仓库当前实现（2025-12-29）。若你后续替换模型或改 API，可按本文的“节点/输入输出”模板同步更新。

---

## 1. 项目概览

本项目是一个**多模态情感识别系统**，主要能力：

- **视频输入**：上传视频后，后端用 ffmpeg 抽取音频与帧，然后分别做：
  - 文本情感（ASR 语音转写 → 文本情感分类）
  - 语音情感（SER：直接从音频做情感分类）
  - 表情情感（FER：从抽帧图像做人脸/表情情绪）
  - 最终将三路结果做融合，输出一个 summary。
- **单模态 API**：支持纯文本、单张图片、单音频的情感识别。
- **训练脚本**：提供文本/音频/图像三套微调脚本，把模型导出到 `models/` 供后端加载。

### 1.1 核心特点

- **后端异步任务**：视频预测走 “提交任务 → 轮询状态 → 获取结果”。
- **模型懒加载**：服务启动时不强制下载模型，首次请求或 warmup 时加载。
- **融合是可替换的**：当前默认用一个“可解释的两层 MLP（固定 7 类）”融合。

---

## 2. 仓库结构（你画架构图时建议的分组）

建议把系统拆成 4 大块：

1) **Frontend（React + Vite）**
- 目录：`frontend/`
- 作用：上传视频、轮询任务、展示融合结果与各模态时间轴

2) **Backend（FastAPI）**
- 目录：`backend/`
- 作用：提供 API、落盘临时文件、调用各模态服务、融合、返回结果

3) **Models（本地模型产物）**
- 目录：`models/`
- 作用：保存训练导出的本地模型（文本/音频/图像/视频）

4) **Data & Scripts（数据与训练/预处理工具）**
- 目录：`data/`、`scripts/`、`download_emotions.py`
- 作用：准备数据、训练、调试加载

> 额外：`robot_zh.py` 是一个**独立的中文情感支持机器人（Flask）**示例服务，和 FastAPI 多模态后端不是同一个进程/链路。

---

## 3. 运行方式与部署形态

### 3.1 Docker Compose（README 提供的一键启动）
- 文件：`docker-compose.yml`
- 服务：
  - `frontend`：Vite dev server，暴露 `3000`
  - `backend`：FastAPI (uvicorn)，暴露 `8000`

前端通过 Vite proxy 将 `/api/*` 转发到 `http://127.0.0.1:8000`。

### 3.2 本地开发（Windows）

- 前端：`scripts/start_frontend.ps1` 可解决端口占用与 PATH 问题
- 后端：建议用 conda 环境安装依赖后启动 uvicorn

> 注意：视频链路强依赖 **ffmpeg**。后端会从 `FFMPEG_PATH` 或系统 PATH 中寻找 ffmpeg。

---

## 4. 对外 API（流程图画“接口节点”的依据）

后端入口文件：`backend/app.py`

### 4.1 视频预测（异步任务）

1) `POST /api/predict/video`
- 入参：`multipart/form-data`，字段名 `file`（视频文件）
- 出参：`{"task_id": "<uuid>"}`
- 副作用：
  - 将视频保存到 `temp/<task_id>.mp4`
  - 写入内存任务表 `tasks[task_id] = {status: processing}`
  - 通过 FastAPI BackgroundTasks 触发后台处理（见 extractor）

2) `GET /api/predict/status/{task_id}`
- 出参：`{"status": "processing|done|error|not_found", ...}`

3) `GET /api/predict/result/{task_id}`
- done 时：返回最终融合结果 JSON
- 非 done：返回 `{"status": "processing|error|not_found"}`

### 4.2 单模态

- `POST /api/predict/text`：JSON body `{"text":"..."}`
- `POST /api/predict/image`：multipart file（单张图片）
- `POST /api/predict/audio`：multipart file（单音频）

### 4.3 模型预热

- `POST /api/models/warmup`
  - 作用：将模型 id 写入环境变量并触发懒加载（ASR/文本情感/语音情感/图像情感）
  - 常用字段：
    - `text_asr_model` → `TEXT_ASR_MODEL`
    - `text_model_id` → `TEXT_MODEL_ID`
    - `audio_model_id` → `AUDIO_MODEL_ID`
    - `image_model_id` → `IMAGE_MODEL_ID`
    - `hf_token` → `HF_TOKEN`
    - 以及 debug/force 开关：`image_force_hf`、`text_debug`、`audio_debug`、`image_debug`

### 4.4 融合调试接口

- `POST /api/fuse_debug`
  - 入参：你可以手工提交 text/audio/image 三个结果 JSON，测试融合策略

---

## 5. 端到端主流程（视频上传 → 多模态 → 融合 → 前端展示）

### 5.1 主流程文字版（适合画总流程图）

1. 用户在前端选择视频
2. 前端 `POST /api/predict/video` 上传
3. 后端保存视频到 `temp/<task_id>.mp4`，立即返回 `task_id`
4. 前端开始轮询 `GET /api/predict/status/<task_id>`
5. 后端后台任务开始：
   1) ffmpeg 从视频抽取音频：`temp/<task_id>.wav`（16kHz/mono）
   2) ffmpeg 抽帧到目录：`temp/<task_id>_frames/frame_0001.jpg ...`（默认 5 fps，可配）
   3) 文本模态：对 `wav` 做 ASR（Whisper）得到 text → 文本情感分类
   4) 音频模态：对 `wav` 做音频情感（Transformers audio-classification 或 SpeechBrain fallback）
   5) 图像模态：对 frames 逐帧做表情情感（DeepFace / fer / HF image-classification）
   6) 融合：把三模态结果送入 fusion_service 的 MLP → 输出 summary + logits
   7) 写入 `tasks[task_id] = {status: done, result: fused_result}`
6. 前端轮询到 done 后，`GET /api/predict/result/<task_id>` 获取结果并渲染时间轴

### 5.2 主流程 Mermaid（可直接复制画图）

```mermaid
flowchart TD
  U[用户选择视频] --> FE[前端: /api/predict/video 上传]
  FE --> BE1[后端: 保存 temp/<task_id>.mp4]
  BE1 --> BE2[返回 task_id]
  BE2 --> POLL[前端: 轮询 /api/predict/status/{task_id}]

  subgraph BG[后端后台任务 extractor.process_video]
    V[输入: temp/<task_id>.mp4] --> A1[ffmpeg 抽音频 -> temp/<task_id>.wav]
    V --> I1[ffmpeg 抽帧 -> temp/<task_id>_frames/]

    A1 --> T1[文本模态: ASR(whisper) -> transcript]
    T1 --> T2[文本情感: text-classification]

    A1 --> S1[语音模态: audio-classification]

    I1 --> F1[图像模态: deepface/fer/HF 逐帧分类]

    T2 --> FU[融合: fusion_service MLP]
    S1 --> FU
    F1 --> FU

    FU --> OUT[写入 tasks[task_id].result]
  end

  POLL -->|done| RES[前端: GET /api/predict/result/{task_id}]
  RES --> UI[前端: Timeline 展示 summary + 各模态时间轴]
```

### 5.3 关键临时文件与“箭头含义”

- 箭头 `视频 -> 音频`：ffmpeg 转码为 16kHz 单声道 wav（提升 ASR/SER 兼容性）
- 箭头 `视频 -> 帧序列`：按 `FRAME_RATE` 抽帧（默认 5fps）
- 箭头 `音频 -> transcript`：ASR（可禁用）
- 箭头 `三模态 -> fusion`：把 time-based scores 聚合后喂入 MLP

---

## 6. 各模态子流程（画子流程图的直接依据）

> 下面每一节都给：输入、处理步骤、输出结构、失败/回退路径。

### 6.1 文本模态（ASR + 文本情感）

实现：`backend/services/text_service.py`

- 输入：`audio_path`（wav）
- 步骤：
  1) `get_asr_pipeline()`：加载 ASR（默认 fallback `openai/whisper-tiny`，可用 `DISABLE_ASR=1` 禁用）
  2) ASR 得到 `text`
  3) `get_text_classifier()`：加载文本情感模型（优先 `TEXT_MODEL_ID`，否则尝试本地 `models/text-emotion`）
  4) `pipeline("text-classification", top_k=None)` 输出各 label+score
  5) 取最高 score label 作为 `summary`

- 输出（典型）：
  - `timeline`: `[{'start':0,'end':0,'text':<transcript>,'emotion':[{'label','score'}, ...]}]`
  - `summary`: `<label>`
  - `valid_text`: bool（用 MIN_TEXT_CHARS/MIN_TEXT_WORDS 判断）
  - 可选 `_debug`

- 失败/回退：
  - ASR 失败 → 返回 neutral + `_debug.reason=asr_failed`
  - 文本模型不可用 → 保留 transcript，但 summary=neutral，并附 `_debug.reason=text_classifier_unavailable`

### 6.2 语音模态（SER）

实现：`backend/services/audio_service.py`

- 输入：`audio_path`
- 步骤：
  1) `_ensure_wav16_mono`：尽量用 torchaudio 转 16k/mono（失败则原样）
  2) `get_audio_classifier()`：
     - 优先 `AUDIO_MODEL_ID`，否则尝试本地 `models/audio-ser`
     - transformers pipeline 失败 → 尝试 SpeechBrain fallback
     - SpeechBrain 也失败 → 尝试备用 HF 模型（默认 `superb/hubert-large-superb-er`）
  3) 调用 pipeline：先“文件路径调用”，失败再尝试“array + sampling_rate”调用
  4) 将 raw 输出归一成 `timeline` 结构
  5) summary=全 timeline 中最高分 label

- 输出（典型）：
  - `timeline`: `[{'start':0,'end':0,'emotion':[{'label','score'}, ...]}]`
  - `summary`: `<label>`
  - 可选 `_debug`（当 `AUDIO_DEBUG=1`）包含 backend、调用路径、异常

- 失败/回退：
  - acoustic classifier 输出空 timeline 时：
    - 若 `AUDIO_ALLOW_TEXT_FALLBACK=1`（默认）→ 回退到 `text_service.predict`，把 text timeline 映射成 audio timeline
    - 可通过 `AUDIO_ALLOW_TEXT_FALLBACK=0` 禁用（训练文档推荐禁用，避免音频=文本）

### 6.3 图像模态（FER：逐帧表情）

实现：`backend/services/image_service.py`

- 输入：`frames_dir`（目录）或 `image_path`（单图）
- 步骤（目录预测）：
  1) 遍历 `frames_dir` 下图片（jpg/png）
  2) `get_image_classifier()`：优先策略：
     - 默认优先 deepface（如果安装）
     - 否则 fer（如果安装）
     - 否则若设置 `IMAGE_MODEL_ID`，使用 HF image-classification pipeline
     - 若 `IMAGE_FORCE_HF=1` 则强制走 HF
  3) 每帧输出 emotion 列表与该帧 summary
  4) overall summary=时间轴里出现最多的 summary

- 输出（典型）：
  - `timeline`: `[{frame:'frame_0001.jpg', emotion:[{label,score}...], summary:'happy'}, ...]`
  - `summary`: `<label>`

- 失败/回退：
  - deepface 需要本地权重文件（避免下载循环）；若缺失会跳过 deepface 尝试 fer/HF
  - 无任何可用后端/模型 → timeline 可能为空，summary=neutral

### 6.4 融合（Multimodal Fusion）

实现：`backend/services/fusion_service.py`

- 固定 7 类顺序：`['angry','disgust','fear','happy','neutral','sad','surprise']`
- 特征：
  - 每个模态从 timeline 聚合得到 7 维“logits-like”分数（累加 score）
  - 再加上每个模态的 duration 与 confidence（最大分）
  - 总输入维度：`7*3 + 6 = 27`
- 模型：2 层 MLP（hidden=64, ReLU）
  - 默认权重是“可解释的确定性权重”（同类 logits 做加和）
  - 也可通过 `FUSION_MLP_WEIGHTS` 指定 JSON 覆盖（w1/b1/w2/b2）

- 输出：
  - `summary`: 融合后的 label
  - `logits`: `{label: score}`（MLP 输出）
  - `timeline`: `{text:..., audio:..., image:...}` 直接携带三模态时间轴
  - `features_used`: 各模态 duration/confidence

---

## 7. 前端交互流程（用于画“UI交互流程图”）

实现：`frontend/src/App.jsx` + `frontend/src/components/Timeline.jsx`

### 7.1 视频上传与轮询

1) 选择视频文件
2) `POST /api/predict/video`，得到 `task_id`
3) 每 1.5 秒轮询 `GET /api/predict/status/{task_id}`
4) status=done 时调用 `GET /api/predict/result/{task_id}`
5) 渲染：
- 顶部显示 `融合结论 summary`
- 展示 `转写文本 transcript`（来自 fused_result.transcript 或 modalities.text.timeline[0].text）
- 三张表分别展示 text/audio/image 的 timeline

### 7.2 文本/图片单模态

- 文本：textarea → `POST /api/predict/text` → 用同一 Timeline 组件展示
- 图片：选择文件 → `POST /api/predict/image` → 展示

---

## 8. 训练与数据流水线（用于画“离线训练流程图”）

训练说明：`TRAINING.md`

### 8.1 文本情感微调

- 数据目录：`txt_Emotional_Dataset/`
  - `train.txt / val.txt / test.txt`
  - 支持多种分隔格式，你当前常见是 `text;label`
- 脚本：`scripts/train_text_hf.py`
- 输出：`models/text-emotion/`
- 后端接入：设置 `TEXT_MODEL_ID=<绝对路径或模型目录>`

### 8.2 语音情感微调（RAVDESS）

- 原始数据：`audio_RAVDESS/Actor_*/...wav`
- 步骤：
  1) 生成 CSV 清单：`scripts/prepare_ravdess.py` → `data/ravdess.csv`
  2) 训练：`scripts/train_audio_hf.py --csv data/ravdess.csv --output_dir models/audio-ser`
- 输出：`models/audio-ser/`
- 后端接入：`AUDIO_MODEL_ID=<...>`，并推荐 `AUDIO_ALLOW_TEXT_FALLBACK=0`

### 8.3 图像表情微调（ImageFolder）

- 数据目录：`img_FER/train/<class>/...` 与 `img_FER/test/<class>/...`
- 脚本：`scripts/train_image_hf.py`
- 输出：`models/image-fer/`
- 后端接入：`IMAGE_MODEL_ID=<...>`（或装 deepface/fer 走本地库）

### 8.4 额外数据下载（HuggingFace datasets）

- 脚本：`download_emotions.py`
- 功能：`datasets.load_dataset("Conna/eMotions")` 并 `save_to_disk("eMotions_local")`
- 产物：`eMotions_local/`（HFDatasets 的本地格式）

---

## 9. 环境变量与配置（流程图旁边的“控制开关”）

### 9.1 运行链路关键开关

- `FFMPEG_PATH`：ffmpeg 可执行文件路径或目录（Windows 兼容）
- `FRAME_RATE`：抽帧 fps（默认 5）

### 9.2 文本/ASR

- `TEXT_ASR_MODEL`：ASR 模型 id（如 `openai/whisper-base`）
- `TEXT_ASR_FALLBACK_MODEL`：未配置时默认 fallback（默认 tiny）
- `DISABLE_ASR=1`：完全禁用 ASR
- `TEXT_MODEL_ID`：文本情感模型（HF id 或本地目录）
- `TEXT_DEBUG=1`：在输出里带 `_debug`
- `MIN_TEXT_CHARS` / `MIN_TEXT_WORDS`：判断 valid_text 的阈值

### 9.3 音频

- `AUDIO_MODEL_ID`：音频情感模型（HF id 或本地目录）
- `AUDIO_BACKUP_HF_ID`：备用 HF 模型（默认 superb/hubert-large-superb-er）
- `AUDIO_ALLOW_TEXT_FALLBACK`：声学模型失败时是否回退到文本（默认 1，训练文档建议 0）
- `AUDIO_DEBUG=1`：输出 `_debug`（包含 raw_info 与调用路径）

### 9.4 图像

- `IMAGE_MODEL_ID`：HF image-classification 模型（或本地目录）
- `IMAGE_FORCE_HF=1`：强制使用 HF pipeline（不走 deepface/fer）
- `IMAGE_DEBUG=1`：单图接口返回 debug

### 9.5 融合

- `FUSION_MLP_WEIGHTS`：融合 MLP 权重 JSON 文件路径

### 9.6 HF

- `HF_TOKEN`：Hugging Face token

---

## 10. 输出数据结构（你画“数据流/消息结构图”时用）

### 10.1 融合结果（视频任务 done 后）

融合结果大体形状：

```json
{
  "summary": "happy",
  "logits": {"angry": 0.1, "...": 0.2},
  "timeline": {
    "text": [{"start": 0, "end": 0, "text": "...", "emotion": [{"label": "joy", "score": 0.9}]}],
    "audio": [{"start": 0, "end": 0, "emotion": [{"label": "happy", "score": 0.8}]}],
    "image": [{"frame": "frame_0001.jpg", "emotion": [{"label": "happy", "score": 0.7}], "summary": "happy"}]
  },
  "features_used": {"text": {"duration": 0, "confidence": 0.9}, "...": {}},
  "transcript": "...",
  "modalities": {"text": {...}, "audio": {...}, "image": {...}}
}
```

> 注意：`modalities` 是 extractor 为了前端展示/排障额外附加的原始输出。

### 10.2 单模态输出

- 文本：`{timeline:[{text,emotion...}], summary}`
- 音频：`{timeline:[{start,end,emotion...}], summary}`
- 图像：`{timeline:[{frame,emotion,summary}], summary}`

---

## 11. 常见故障与排障节点（建议画在流程图的“异常分支”上）

1) **ffmpeg 找不到**
- 现象：视频任务立刻 error，错误包含 `ffmpeg not found in PATH`
- 处理：安装 ffmpeg，并设置 `FFMPEG_PATH` 或加入系统 PATH

2) **ASR 转写为空**
- 现象：前端“转写文本”为空，Timeline 提示“未启用 ASR 或转写失败”
- 处理：
  - 检查 `TEXT_ASR_MODEL` 是否可用（或使用默认 tiny）
  - 开启 `TEXT_DEBUG=1` 查看 `_debug.reason`

3) **文本/音频/图像模型未配置**
- 现象：某模态 timeline 为空或 summary=neutral
- 处理：用 `/api/models/warmup` 指定本地模型目录或 HF 模型 id

4) **图像 deepface 权重缺失**
- 现象：deepface 安装了但识别为空/报错；单图接口可能提示跳过 deepface
- 处理：确保 deepface 权重存在（或设置 `IMAGE_FORCE_HF=1` 走 HF 模型）

5) **temp 目录膨胀**
- 现象：大量 `temp/<task_id>_frames` 与 wav/mp4 残留
- 建议：生产应定期清理 temp（本项目当前 extractor 未自动清理视频任务产物）

---

## 12. 流程图绘制清单（直接照抄节点名）

### 12.1 总体架构图节点（模块级）

- Frontend (React/Vite)
- Backend API (FastAPI)
- Extractor (ffmpeg)
- Text Service (ASR + Text Emotion)
- Audio Service (SER)
- Image Service (FER)
- Fusion Service (MLP)
- Local Models (models/)
- Temp Storage (temp/)

### 12.2 视频预测流程图节点（详细级）

建议节点顺序：

1. 上传视频（前端）
2. 创建 task_id（后端）
3. 保存 mp4 到 temp
4. 后台任务启动
5. ffmpeg 抽音频 wav
6. ffmpeg 抽帧 frames
7. 文本：ASR→transcript
8. 文本：text-classification
9. 音频：audio-classification（含 fallback）
10. 图像：逐帧分类（deepface/fer/HF）
11. 融合：MLP
12. 保存结果到 tasks
13. 前端轮询 status
14. 前端拉取 result
15. 前端展示 timeline

### 12.3 单模态流程图节点

- 文本：输入文本 → 文本模型 → summary/timeline
- 音频：上传音频 → 规范化采样率 → 音频模型 → summary/timeline（可选回退文本）
- 图片：上传图片 → 选择后端（deepface/fer/HF） → summary/timeline

### 12.4 离线训练流程图节点

- 数据准备：
  - 文本：解析 train/val/test
  - 音频：扫描 RAVDESS → 生成 ravdess.csv
  - 图像：ImageFolder 目录结构
- 训练：加载 base_model → 训练循环 → 保存到 models/<modality>
- 部署接入：设置 env / warmup → 后端加载

---

## 13. 关键文件索引（定位实现时用）

- 后端 API：`backend/app.py`
- 视频处理 orchestrator：`backend/services/extractor.py`
- 文本服务：`backend/services/text_service.py`
- 音频服务：`backend/services/audio_service.py`
- 图像服务：`backend/services/image_service.py`
- 融合服务：`backend/services/fusion_service.py`
- 前端主界面：`frontend/src/App.jsx`
- 前端结果表格：`frontend/src/components/Timeline.jsx`
- 前端启动脚本：`scripts/start_frontend.ps1`
- 训练文档：`TRAINING.md`
- 文本训练脚本：`scripts/train_text_hf.py`
- 音频训练脚本：`scripts/train_audio_hf.py`
- 图像训练脚本：`scripts/train_image_hf.py`
- RAVDESS 清单生成：`scripts/prepare_ravdess.py`
- eMotions 下载：`download_emotions.py`
- 中文情感支持机器人（独立）：`robot_zh.py`
