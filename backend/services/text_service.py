"""Text emotion service.

- ASR: optional (`TEXT_ASR_MODEL`)
- Text emotion classifier: optional (`TEXT_MODEL_ID`)

Both can point to a Hugging Face model id or a local directory.
"""

import logging
import os
from typing import Any, Dict, List, Optional


def _project_root() -> str:
    # backend/services -> project root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _local_model_dir(*parts: str) -> Optional[str]:
    p = os.path.join(_project_root(), *parts)
    return p if os.path.isdir(p) else None

logger = logging.getLogger(__name__)

_asr = None
_text_classifier = None
_asr_load_error: Optional[str] = None
_text_load_error: Optional[str] = None


def _text_debug_enabled() -> bool:
    return os.environ.get("TEXT_DEBUG", "0") == "1"


def get_asr_pipeline(model_id: Optional[str] = None):
    global _asr
    global _asr_load_error
    if model_id is None:
        model_id = os.environ.get("TEXT_ASR_MODEL")
        # Default-enable ASR for video if not configured.
        # Set DISABLE_ASR=1 to fully disable, or set TEXT_ASR_MODEL to a specific model.
        if (not model_id) and os.environ.get("DISABLE_ASR", "0") != "1":
            model_id = os.environ.get("TEXT_ASR_FALLBACK_MODEL", "openai/whisper-tiny")
    if not model_id:
        return None
    if _asr is None:
        try:
            from transformers import pipeline

            _asr = pipeline("automatic-speech-recognition", model=model_id)
            _asr_load_error = None
        except Exception as e:
            logger.warning(f"Failed to load ASR pipeline ({model_id}): {e}")
            _asr = None
            _asr_load_error = str(e)
    return _asr


def get_text_classifier(model_id: Optional[str] = None):
    global _text_classifier
    global _text_load_error
    if model_id is None:
        model_id = os.environ.get("TEXT_MODEL_ID")
        # Default to local model if present, to avoid silent neutral/empty outputs.
        if not model_id:
            model_id = _local_model_dir("models", "text-emotion")
    if not model_id:
        return None
    if _text_classifier is None:
        try:
            from transformers import pipeline

            _text_classifier = pipeline("text-classification", model=model_id, top_k=None)
            _text_load_error = None
        except Exception as e:
            logger.warning(f"Failed to load text classifier ({model_id}): {e}")
            _text_classifier = None
            _text_load_error = str(e)
    return _text_classifier


def _flatten_scores(emotion_scores: Any) -> List[Dict[str, Any]]:
    flat: List[Dict[str, Any]] = []
    if isinstance(emotion_scores, list):
        for item in emotion_scores:
            if isinstance(item, list):
                for x in item:
                    if isinstance(x, dict):
                        flat.append(x)
            elif isinstance(item, dict):
                flat.append(item)
    elif isinstance(emotion_scores, dict):
        flat.append(emotion_scores)
    return flat


def _is_valid_text(text: str) -> bool:
    min_chars = int(os.environ.get("MIN_TEXT_CHARS", "5"))
    min_words = int(os.environ.get("MIN_TEXT_WORDS", "2"))
    words = [w for w in text.strip().split() if any(c.isalnum() for c in w)]
    is_valid = len(text.strip()) >= min_chars and len(words) >= min_words
    invalid_tokens = {"uh", "um", "you", "yeah", "mm"}
    if len(words) == 1 and words[0].lower() in invalid_tokens:
        is_valid = False
    return is_valid


def predict(audio_path: str) -> dict:
    """ASR -> text emotion.

    Returns:
      {'timeline': [...], 'summary': label, 'valid_text': bool}
    """
    asr = get_asr_pipeline()
    clf = get_text_classifier()

    try:
        result = asr(audio_path)
        text = (result or {}).get("text", "")
    except Exception as e:
        logger.warning(f"ASR failed: {e}")
        fail = {"timeline": [], "summary": "neutral", "valid_text": False}
        if _text_debug_enabled() or os.environ.get("DEBUG_ON_FAILURE", "1") == "1":
            fail["_debug"] = {
                "reason": "asr_failed",
                "error": str(e),
                "text_asr_model": os.environ.get("TEXT_ASR_MODEL"),
            }
        return fail

    if clf is None:
        out = {
            "timeline": [{"start": 0, "end": 0, "text": text, "emotion": []}],
            "summary": "neutral",
            "valid_text": _is_valid_text(text),
        }
        if _text_debug_enabled() or os.environ.get("DEBUG_ON_FAILURE", "1") == "1":
            out["_debug"] = {
                "reason": "text_classifier_unavailable",
                "text_model_id": os.environ.get("TEXT_MODEL_ID") or _local_model_dir("models", "text-emotion"),
                "asr_text_len": len(text or ""),
            }
        return out

    try:
        emotion_scores = clf(text)
        flat = _flatten_scores(emotion_scores)
        summary = max(flat, key=lambda x: x.get("score", 0)).get("label", "neutral") if flat else "neutral"
        out = {
            "timeline": [{"start": 0, "end": 0, "text": text, "emotion": flat}],
            "summary": summary,
            "valid_text": _is_valid_text(text),
        }
        if _text_debug_enabled():
            out["_debug"] = {
                "reason": "ok",
                "text_model_id": os.environ.get("TEXT_MODEL_ID") or _local_model_dir("models", "text-emotion"),
                "asr_text_len": len(text or ""),
            }
        # If ASR produced empty text, surface that for diagnosis.
        elif os.environ.get("DEBUG_ON_FAILURE", "1") == "1" and not (text or "").strip():
            out["_debug"] = {
                "reason": "asr_returned_empty_text",
                "text_asr_model": os.environ.get("TEXT_ASR_MODEL"),
            }
        return out
    except Exception as e:
        logger.warning(f"Text classification failed: {e}")
        out = {
            "timeline": [{"start": 0, "end": 0, "text": text, "emotion": []}],
            "summary": "neutral",
            "valid_text": _is_valid_text(text),
        }
        if _text_debug_enabled() or os.environ.get("DEBUG_ON_FAILURE", "1") == "1":
            out["_debug"] = {
                "reason": "text_classification_failed",
                "error": str(e),
                "text_model_id": os.environ.get("TEXT_MODEL_ID") or _local_model_dir("models", "text-emotion"),
                "asr_text_len": len(text or ""),
            }
        return out


def predict_from_text(text: str) -> dict:
    clf = get_text_classifier()
    if clf is None:
        return {"timeline": [{"start": 0, "end": 0, "text": text, "emotion": []}], "summary": "neutral"}

    try:
        emotion_scores = clf(text)
        flat = _flatten_scores(emotion_scores)
        summary = max(flat, key=lambda x: x.get("score", 0)).get("label", "neutral") if flat else "neutral"
        return {"timeline": [{"start": 0, "end": 0, "text": text, "emotion": flat}], "summary": summary}
    except Exception as e:
        logger.warning(f"Text classification failed: {e}")
        return {"timeline": [{"start": 0, "end": 0, "text": text, "emotion": []}], "summary": "neutral"}
