# 图像表情识别服务（人脸检测+表情分类） - 延迟加载并容错
import logging
import os
from PIL import Image

logger = logging.getLogger(__name__)

_image_classifier = None

def get_image_classifier():
    global _image_classifier
    if _image_classifier is None:
        try:
            from transformers import pipeline
            model_id = os.environ.get('IMAGE_MODEL_ID')
            if model_id is None:
                # 未配置模型 id，保持 None（不自动下载）
                return None
            _image_classifier = pipeline('image-classification', model=model_id)
        except Exception as e:
            logger.warning(f"Failed to load image classifier: {e}")
            _image_classifier = None
    return _image_classifier
import logging
import os
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

_image_classifier = None

def _use_deepface(img):
    try:
        from deepface import DeepFace
        # DeepFace.analyze expects a numpy array (BGR) or path
        res = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        # res can be dict or list
        if isinstance(res, list):
            res = res[0]
        emotions = res.get('emotion', {})
        # convert to list of {label, score}
        return [{'label': k, 'score': float(v)} for k, v in emotions.items()]
    except Exception as e:
        logger.exception(f"deepface not available or failed: {e}")
        return None

def _use_fer(img):
    try:
        from fer import FER
        detector = FER(mtcnn=True)
        # FER expects RGB ndarray
        arr = np.array(img)
        results = detector.detect_emotions(arr)
        if not results:
            return []
        # take first face
        emotions = results[0].get('emotions', {})
        return [{'label': k, 'score': float(v)} for k, v in emotions.items()]
    except Exception as e:
        logger.debug(f"fer not available or failed: {e}")
        return None

def get_image_classifier():
    """Return a callable classifier or None. We prefer local libs (deepface, fer).
    If IMAGE_MODEL_ID is set, we will attempt to use transformers pipeline as fallback.
    """
    global _image_classifier
    if _image_classifier is not None:
        return _image_classifier

    # Optional: force using HF/local model directory for validation
    if os.environ.get('IMAGE_FORCE_HF', '0') == '1':
        model_id = os.environ.get('IMAGE_MODEL_ID')
        if model_id:
            try:
                from transformers import pipeline
                pipe = pipeline('image-classification', model=model_id)
                _image_classifier = pipe
                return _image_classifier
            except Exception as e:
                logger.warning(f"Failed to load forced HF image classifier ({model_id}): {e}")
                _image_classifier = None
        return None

    # Prefer deepface
    try:
        import deepface  # noqa: F401
        _image_classifier = 'deepface'
        return _image_classifier
    except Exception:
        pass

    # Then fer
    try:
        import fer  # noqa: F401
        _image_classifier = 'fer'
        return _image_classifier
    except Exception:
        pass

    # Finally try transformers if IMAGE_MODEL_ID specified
    model_id = os.environ.get('IMAGE_MODEL_ID')
    if model_id:
        try:
            from transformers import pipeline
            pipe = pipeline('image-classification', model=model_id)
            _image_classifier = pipe
            return _image_classifier
        except Exception as e:
            logger.warning(f"Failed to load HF image classifier ({model_id}): {e}")
            _image_classifier = None

    return None

def _classify_with_hf(pipe, img):
    try:
        return pipe(img)
    except Exception as e:
        logger.debug(f"HF image classifier failed: {e}")
        return None

def predict(frames_dir: str) -> dict:
    """Process frames in a directory and return per-frame emotion scores and an overall summary.
    Uses deepface or fer if installed; otherwise uses HF model if IMAGE_MODEL_ID is set; else returns neutral.
    """
    classifier = get_image_classifier()
    timeline = []
    for fname in sorted(os.listdir(frames_dir)):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        frame_path = os.path.join(frames_dir, fname)
        try:
            img = Image.open(frame_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Failed to open image {frame_path}: {e}")
            continue

        emotion_list = []
        summary = 'neutral'

        if classifier == 'deepface':
            res = _use_deepface(frame_path)
            if res:
                emotion_list = res
                summary = max(emotion_list, key=lambda x: x['score'])['label']
        elif classifier == 'fer':
            res = _use_fer(img)
            if res:
                emotion_list = res
                summary = max(emotion_list, key=lambda x: x['score'])['label']
        elif callable(classifier):
            res = _classify_with_hf(classifier, img)
            if res:
                # HF returns list of dicts with label and score
                emotion_list = [{'label': r.get('label'), 'score': float(r.get('score', 0))} for r in res]
                try:
                    summary = max(emotion_list, key=lambda x: x['score'])['label']
                except Exception:
                    summary = 'neutral'

        timeline.append({'frame': fname, 'emotion': emotion_list, 'summary': summary})

    # overall summary: most common non-neutral label
    from collections import Counter
    labels = [t['summary'] for t in timeline if t['summary']]
    if labels:
        most = Counter(labels).most_common(1)
        overall = most[0][0]
    else:
        overall = 'neutral'

    return {'timeline': timeline, 'summary': overall}


def predict_image(image_path: str) -> dict:
    """对单张图像文件进行情感（表情）识别，返回与多帧一致的格式，timeline 包含一条记录。"""
    classifier = get_image_classifier()
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        logger.warning(f"Failed to open image {image_path}: {e}")
        return {'timeline': [], 'summary': 'neutral'}

    emotion_list = []
    summary = 'neutral'
    debug_info = {}
    if os.environ.get('IMAGE_DEBUG') == '1':
        debug_info['backend'] = 'unknown'
        debug_info['image_model_id'] = os.environ.get('IMAGE_MODEL_ID')
        debug_info['image_force_hf'] = os.environ.get('IMAGE_FORCE_HF', '0')

    # helper: check deepface weight file exists/size
    def _deepface_weights_ok():
        try:
            w = os.path.join(os.path.expanduser('~'), '.deepface', 'weights', 'facial_expression_model_weights.h5')
            if os.path.exists(w) and os.path.getsize(w) > 5_000_000:
                return True
        except Exception:
            pass
        return False

    if classifier == 'deepface':
        if os.environ.get('IMAGE_DEBUG') == '1':
            debug_info['backend'] = 'deepface'
        # avoid triggering DeepFace download loop: require local weights presence or allow fallback
        if not _deepface_weights_ok():
            logger.warning('DeepFace weights missing or too small; skipping DeepFace and trying fer/HF fallback')
            debug_info['deepface_skipped'] = True
        else:
            try:
                res = _use_deepface(image_path)
                debug_info['deepface_raw'] = res
                if res:
                    emotion_list = res
                    summary = max(emotion_list, key=lambda x: x['score'])['label']
                else:
                    debug_info['deepface_empty'] = True
            except Exception as e:
                logger.exception(f"DeepFace analyze failed for {image_path}: {e}")
                debug_info['deepface_exception'] = str(e)
    elif classifier == 'fer':
        if os.environ.get('IMAGE_DEBUG') == '1':
            debug_info['backend'] = 'fer'
        res = _use_fer(img)
        if res:
            emotion_list = res
            summary = max(emotion_list, key=lambda x: x['score'])['label']
    elif callable(classifier):
        if os.environ.get('IMAGE_DEBUG') == '1':
            debug_info['backend'] = 'hf'
        res = _classify_with_hf(classifier, img)
        if res:
            emotion_list = [{'label': r.get('label'), 'score': float(r.get('score', 0))} for r in res]
            try:
                summary = max(emotion_list, key=lambda x: x['score'])['label']
            except Exception:
                summary = 'neutral'

    timeline_entry = {'frame': os.path.basename(image_path), 'emotion': emotion_list, 'summary': summary}
    # include debug info if requested via env var
    if os.environ.get('IMAGE_DEBUG') == '1':
        timeline_entry['debug'] = debug_info

    timeline = [timeline_entry]
    result = {'timeline': timeline, 'summary': summary}

    # If DeepFace produced nothing and we have fer available, try fallback now
    if (not emotion_list) and debug_info.get('deepface_empty') or debug_info.get('deepface_skipped'):
        try:
            # try fer if available
            if 'fer' in globals() or 'fer' in locals():
                res2 = _use_fer(img)
            else:
                res2 = _use_fer(img)
            if res2:
                timeline[0]['emotion'] = res2
                timeline[0]['summary'] = max(res2, key=lambda x: x['score'])['label']
                result = {'timeline': timeline, 'summary': timeline[0]['summary']}
        except Exception:
            pass

    return result

