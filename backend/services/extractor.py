# 视频/音频/帧提取服务
import os
import subprocess
import shutil
import logging

logger = logging.getLogger(__name__)

def process_video(video_path, task_id, tasks):
    # 用 ffmpeg 提取音频和帧
    audio_path = f"temp/{task_id}.wav"
    frames_dir = f"temp/{task_id}_frames"
    os.makedirs(frames_dir, exist_ok=True)

    # 优先使用环境变量 FFMPEG_PATH（可指向可执行文件），否则在 PATH 中查找 ffmpeg
    ffmpeg_path = os.environ.get('FFMPEG_PATH') or shutil.which("ffmpeg")
    if ffmpeg_path:
        # 在 Windows 用户可能传入目录或可执行完整路径，处理这两种情况
        if os.path.isdir(ffmpeg_path):
            candidate = os.path.join(ffmpeg_path, 'ffmpeg.exe' if os.name == 'nt' else 'ffmpeg')
            if os.path.exists(candidate):
                ffmpeg_path = candidate
        elif not os.path.exists(ffmpeg_path):
            # 如果用户传入不存在的路径，置为 None 以触发错误处理
            ffmpeg_path = None
    
    # 如果仍然找不到 ffmpeg，返回错误并记录
    if ffmpeg_path is None:
        err = "ffmpeg not found in PATH. Please install ffmpeg and add to PATH."
        logger.error(err)
        tasks[task_id] = {"status": "error", "error": err}
        return

    # 提取音频
    try:
        # 统一导出为 16kHz 单声道 wav，提升 ASR/SER 兼容性
        subprocess.run(
            [
                ffmpeg_path,
                "-y",
                "-i",
                video_path,
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                audio_path,
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg audio extraction failed: {e}")
        tasks[task_id] = {"status": "error", "error": "ffmpeg audio extraction failed"}
        return

    # 提取帧：帧率可通过环境变量 `FRAME_RATE` 配置（默认 5 fps）
    frame_rate = int(os.environ.get('FRAME_RATE', '5'))
    out_pattern = os.path.join(frames_dir, 'frame_%04d.jpg')
    try:
        subprocess.run([ffmpeg_path, "-y", "-i", video_path, "-vf", f"fps={frame_rate}", out_pattern], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg frame extraction failed: {e}")
        tasks[task_id] = {"status": "error", "error": "ffmpeg frame extraction failed"}
        return
    # 统计提取到的帧数，便于诊断
    try:
        frames = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        logger.info(f"Extracted {len(frames)} frames at {frame_rate} fps into {frames_dir}")
    except Exception:
        logger.debug("Could not list frames after extraction")
    # 延迟导入并调用各模态服务（已在各服务中实现延迟加载）
    try:
        from . import text_service, audio_service, image_service, fusion_service
        text_result = text_service.predict(audio_path)
        audio_result = audio_service.predict(audio_path)
        image_result = image_service.predict(frames_dir)
        fused_result = fusion_service.fuse(text_result, audio_result, image_result)
        # 兼容前端展示与排障：附带各模态原始输出（含 _debug）以及 transcript 便捷字段
        transcript = ''
        try:
            tl = text_result.get('timeline') if isinstance(text_result, dict) else None
            if isinstance(tl, list) and tl:
                transcript = str(tl[0].get('text') or '')
        except Exception:
            transcript = ''

        fused_result['transcript'] = transcript
        fused_result['modalities'] = {
            'text': text_result,
            'audio': audio_result,
            'image': image_result,
        }
        tasks[task_id] = {"status": "done", "result": fused_result}
    except Exception as e:
        logger.exception(f"Error during multimodal prediction: {e}")
        tasks[task_id] = {"status": "error", "error": str(e)}
