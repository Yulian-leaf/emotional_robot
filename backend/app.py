# FastAPI 主入口，多模态情感识别后端
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
import uuid
import os
from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import FileResponse
from typing import Any, Dict

# 为避免在导入时触发模型下载，service 在需要时再导入（懒加载）


app = FastAPI()

# 任务存储（简单内存，生产建议用 Redis）
tasks = {}

@app.post("/api/predict/video")
async def predict_video(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    task_id = str(uuid.uuid4())
    # 保存上传文件到临时目录
    os.makedirs("temp", exist_ok=True)
    save_path = f"temp/{task_id}.mp4"
    with open(save_path, "wb") as f:
        f.write(await file.read())
    tasks[task_id] = {"status": "processing"}
    # 延迟导入 services，避免启动时下载模型
    from .services import extractor
    if background_tasks is not None:
        background_tasks.add_task(extractor.process_video, save_path, task_id, tasks)
    else:
        # 如果没有 BackgroundTasks（极少发生），同步执行任务
        extractor.process_video(save_path, task_id, tasks)
    return {"task_id": task_id}


@app.post('/api/predict/text')
async def predict_text(payload: dict):
    """纯文本情感预测：JSON body {"text": "..."} 返回 text_service.predict_from_text 结果。"""
    text = payload.get('text') if isinstance(payload, dict) else None
    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text' in request body")
    from .services import text_service
    res = text_service.predict_from_text(text)
    return JSONResponse(res)


@app.post('/api/predict/image')
async def predict_image(file: UploadFile = File(...)):
    """单张图片情感预测：接收 multipart file，返回 image_service.predict_image 结果。"""
    os.makedirs('temp', exist_ok=True)
    task_id = str(uuid.uuid4())
    save_path = f"temp/{task_id}_{file.filename}"
    with open(save_path, 'wb') as f:
        f.write(await file.read())
    from .services import image_service
    res = image_service.predict_image(save_path)
    # 可选择删除临时文件
    try:
        os.remove(save_path)
    except Exception:
        pass
    return JSONResponse(res)


@app.post('/api/predict/audio')
async def predict_audio(file: UploadFile = File(...)):
    """单音频文件情感预测：接收 multipart file，返回 audio_service.predict 结果。"""
    os.makedirs('temp', exist_ok=True)
    task_id = str(uuid.uuid4())
    save_path = f"temp/{task_id}_{file.filename}"
    with open(save_path, 'wb') as f:
        f.write(await file.read())
    from .services import audio_service
    res = audio_service.predict(save_path)
    try:
        os.remove(save_path)
    except Exception:
        pass
    return JSONResponse(res)


@app.post('/api/fuse_debug')
async def fuse_debug(payload: Dict[str, Any]):
    """调试用接口：提交 text_result/audio_result/image_result JSON，并可选传入 weights/strategy。
    返回各策略下的 score_map、选定 summary 以及用于诊断的细节。
    示例 body: {"text_result": {...}, "audio_result": {...}, "image_result": {...}, "weights": {"text":1,"audio":1,"image":2}, "recent_frames": 10}
    """
    try:
        from .services import fusion_service
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"fusion_service import failed: {e}")

    text_result = payload.get('text_result', {})
    audio_result = payload.get('audio_result', {})
    image_result = payload.get('image_result', {})
    weights = payload.get('weights')
    recent_n = int(payload.get('recent_frames', 8))

    # Aggregate strategy (uses fusion_service.fuse)
    agg = fusion_service.fuse(text_result, audio_result, image_result, weights=weights)

    # Majority strategy: count per-modality summaries and image per-frame summaries
    def majority_label(tr):
        if not tr:
            return 'neutral'
        if isinstance(tr, dict):
            tl = tr.get('timeline', [])
            labels = []
            for seg in tl:
                if 'summary' in seg:
                    labels.append(seg['summary'])
                elif 'emotion' in seg and seg['emotion']:
                    # pick highest in emotion list
                    labels.append(max(seg['emotion'], key=lambda x: x.get('score', 0)).get('label'))
            if labels:
                from collections import Counter
                return Counter(labels).most_common(1)[0][0]
        return tr.get('summary', 'neutral') if isinstance(tr, dict) else 'neutral'

    maj_text = majority_label(text_result)
    maj_audio = majority_label(audio_result)
    maj_image = majority_label(image_result)
    from collections import Counter
    maj_votes = [maj_text, maj_audio, maj_image]
    maj_summary = Counter(maj_votes).most_common(1)[0][0]

    # Recent-window aggregate: only consider last `recent_n` image frames
    img_tl = image_result.get('timeline', []) if isinstance(image_result, dict) else []
    recent_img = img_tl[-recent_n:] if img_tl else []
    recent_payload = {'timeline': text_result.get('timeline', []), 'audio': audio_result.get('timeline', []), 'image': recent_img}
    recent_agg = fusion_service.fuse(text_result, audio_result, {'timeline': recent_img, 'summary': image_result.get('summary') if isinstance(image_result, dict) else 'neutral'}, weights=weights)

    return JSONResponse({
        'aggregate': agg,
        'majority': {'text': maj_text, 'audio': maj_audio, 'image': maj_image, 'summary': maj_summary},
        'recent': {'recent_frames': recent_n, 'result': recent_agg}
    })

@app.get("/api/predict/status/{task_id}")
def get_status(task_id: str):
    return tasks.get(task_id, {"status": "not_found"})

@app.get("/api/predict/result/{task_id}")
def get_result(task_id: str):
    result = tasks.get(task_id, {})
    if result.get("status") == "done":
        return JSONResponse(result["result"])
    return {"status": result.get("status", "not_found")}


@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs"}


@app.post('/api/models/warmup')
def warmup_models(payload: dict, background_tasks: BackgroundTasks = None):
    """按需预热模型。请求体 JSON 支持字段：
    - text_asr_model: 模型 id（如 openai/whisper-base）
    - text_model_id: 文本情感模型 id
    - audio_model_id: 语音情感模型 id
    - image_model_id: 图像模型 id
    - hf_token: 可选 Hugging Face token（将设置 HF_TOKEN 环境变量）
    """
    # 将模型 id 写入环境变量以供服务读取
    for k, envname in (('text_asr_model', 'TEXT_ASR_MODEL'), ('text_model_id', 'TEXT_MODEL_ID'), ('audio_model_id', 'AUDIO_MODEL_ID'), ('image_model_id', 'IMAGE_MODEL_ID')):
        if k in payload:
            os.environ[envname] = payload[k]
    # optional flags for validation/debug
    for k, envname in (
        ('image_force_hf', 'IMAGE_FORCE_HF'),
        ('image_debug', 'IMAGE_DEBUG'),
        ('audio_debug', 'AUDIO_DEBUG'),
        ('text_debug', 'TEXT_DEBUG'),
    ):
        if k in payload:
            os.environ[envname] = str(payload[k])
    if 'hf_token' in payload:
        os.environ['HF_TOKEN'] = payload['hf_token']

    def _load():
        # 尝试导入并触发延迟加载
        try:
            from .services import text_service, audio_service, image_service
            text_service.get_asr_pipeline()
            text_service.get_text_classifier()
            audio_service.get_audio_classifier()
            image_service.get_image_classifier()
        except Exception as e:
            # 捕获异常，记录但不抛出
            import logging
            logging.getLogger(__name__).exception(f"Warmup failed: {e}")

    if background_tasks is not None:
        background_tasks.add_task(_load)
        return {"status": "warming_up"}
    else:
        _load()
        return {"status": "done"}

# ...可扩展：模型微调、单模态API、融合API等
