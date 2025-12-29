"""语音情感识别服务（SER）
改进点：
- 延迟加载模型（通过环境变量 `AUDIO_MODEL_ID` 指定）
- 在必要时对音频进行重采样/合并为单声道以保证一致性
- 提供 `warmup` 支持并改进错误处理与返回结构
"""
import logging
import os
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)

_audio_classifier = None
_audio_model_id = None
_audio_device = None
_audio_backend = None
_audio_load_error = None


def _project_root() -> str:
    # backend/services -> project root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _local_model_dir(*parts: str) -> Optional[str]:
    p = os.path.join(_project_root(), *parts)
    return p if os.path.isdir(p) else None


def _text_result_to_audio_timeline(text_result: dict):
    """将 text_service.predict 的输出转换为 audio timeline 结构。

    text_result.timeline: [{'start','end','text','emotion':[{'label','score'}]}]
    audio timeline 期望:  [{'start','end','emotion':[...]}]
    """
    if not isinstance(text_result, dict):
        return []
    tl = text_result.get('timeline')
    if not isinstance(tl, list) or not tl:
        return []
    out = []
    for seg in tl:
        if not isinstance(seg, dict):
            continue
        emos = seg.get('emotion')
        # 允许空 emotion，但不要把 text 字段带进 audio
        if not isinstance(emos, list):
            emos = []
        out.append({
            'start': float(seg.get('start', 0) or 0),
            'end': float(seg.get('end', 0) or 0),
            'emotion': [
                {'label': e.get('label'), 'score': float(e.get('score', 0) or 0)}
                for e in emos
                if isinstance(e, dict)
            ],
        })
    return out

def get_audio_classifier(model_id: Optional[str] = None):
    """懒加载 audio-classification pipeline。返回 pipeline 或 None。
    如果可用，优先使用环境变量 `AUDIO_MODEL_ID`。
    """
    global _audio_classifier
    global _audio_model_id, _audio_device, _audio_backend, _audio_load_error
    if model_id is None:
        model_id = os.environ.get('AUDIO_MODEL_ID')
        # Prefer local model to avoid silent fallback to text.
        if not model_id:
            model_id = _local_model_dir('models', 'audio-ser')
        # Optional: allow using backup model even when unset (may download).
        if (not model_id) and os.environ.get('AUDIO_USE_BACKUP_IF_UNSET', '0') == '1':
            model_id = os.environ.get('AUDIO_BACKUP_HF_ID', 'superb/hubert-large-superb-er')

    if not model_id:
        return None

    if _audio_classifier is not None:
        return _audio_classifier

    try:
        from transformers import pipeline
        try:
            import torch
            device = 0 if torch.cuda.is_available() else -1
        except Exception:
            device = -1
        _audio_classifier = pipeline('audio-classification', model=model_id, device=device)
        _audio_model_id = model_id
        _audio_device = device
        _audio_backend = 'transformers'
        logger.info(f"Loaded audio classifier: {model_id} (device={device})")
    except Exception as e:
        logger.warning(f"Failed to load audio classifier ({model_id}): {e}")
        _audio_classifier = None
        _audio_model_id = model_id
        try:
            _audio_load_error = str(e)
        except Exception:
            _audio_load_error = repr(e)
        _audio_device = None
        # 尝试 SpeechBrain 作为后备
        try:
            import importlib
            try:
                sb_pkg = importlib.import_module('speechbrain.pretrained')
            except Exception:
                sb_pkg = importlib.import_module('speechbrain.inference')
            EncoderClassifier = getattr(sb_pkg, 'EncoderClassifier')
            sb_model_id = model_id if model_id and model_id.startswith('speechbrain/') else 'speechbrain/emotion-recognition-wav2vec2-IEMOCAP'
            try:
                import torch
                run_opts = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
            except Exception:
                run_opts = {}
            sb = EncoderClassifier.from_hparams(source=sb_model_id, savedir=None, run_opts=run_opts)
            _audio_classifier = sb
            _audio_backend = 'speechbrain'
            _audio_load_error = None
            logger.info(f"Loaded SpeechBrain audio classifier: {sb_model_id}")
        except Exception as e2:
            logger.warning(f"SpeechBrain fallback failed: {e2}")
            try:
                _audio_load_error = str(e) + ' | speechbrain_error: ' + str(e2)
            except Exception:
                _audio_load_error = repr((e, e2))
            _audio_backend = None
            # 再尝试一个已知可用的 transformers 备用模型，避免 audio timeline 为空
            backup_id = os.environ.get('AUDIO_BACKUP_HF_ID', 'superb/hubert-large-superb-er')
            try:
                from transformers import pipeline as hf_pipeline
                try:
                    import torch
                    device = 0 if torch.cuda.is_available() else -1
                except Exception:
                    device = -1
                _audio_classifier = hf_pipeline('audio-classification', model=backup_id, device=device)
                _audio_model_id = backup_id
                _audio_device = device
                _audio_backend = 'transformers'
                logger.info(f"Loaded backup HF audio classifier: {backup_id} (device={device})")
            except Exception as e3:
                logger.warning(f"Backup HF model load failed ({backup_id}): {e3}")
                try:
                    _audio_load_error = _audio_load_error + ' | backup_error: ' + str(e3)
                except Exception:
                    pass

    return _audio_classifier


def warmup_audio_model(model_id: Optional[str] = None):
    """Explicit预热模型（用于 `/api/models/warmup` 调用）"""
    return get_audio_classifier(model_id=model_id) is not None


def _ensure_wav16_mono(src_path: str) -> str:
    """将任意音频转换为 16kHz 单声道 WAV 并返回新路径（临时文件）。
    使用 torchaudio 加载/重采样，若 torchaudio 不可用则直接返回原路径（pipeline 可能支持文件）。
    """
    try:
        import torchaudio
        waveform, sr = torchaudio.load(src_path)
        # 合并为单声道
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        target_sr = 16000
        if sr != target_sr:
            try:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                waveform = resampler(waveform)
                sr = target_sr
            except Exception as e:
                logger.debug(f"Resample failed: {e}")

        # 保存为临时 wav 文件
        fd, tmp_path = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
        torchaudio.save(tmp_path, waveform, sr)
        return tmp_path
    except Exception as e:
        logger.debug(f"torchaudio not available or failed to process audio: {e}")
        return src_path


def predict(audio_path: str) -> dict:
    """对单个音频文件进行情感预测。
    返回结构：{'timeline': [{'start': float, 'end': float, 'emotion': [...]}, ...], 'summary': str}
    如果模型不可用或发生错误，返回 neutral。
    """
    classifier = get_audio_classifier()

    in_path = audio_path
    tmp_path = None
    try:
        # 先尝试将音频规范化为 16k mono WAV，提升兼容性
        tmp_path = _ensure_wav16_mono(audio_path)
        if tmp_path != audio_path:
            in_path = tmp_path
        # 如果后端是 SpeechBrain，我们使用专用调用流程并返回
        if _audio_backend == 'speechbrain':
            try:
                # speechbrain 需要文件路径（已经规范化为 in_path）
                raw = None
                try:
                    if hasattr(_audio_classifier, 'predict_file'):
                        raw = _audio_classifier.predict_file(in_path)
                    elif hasattr(_audio_classifier, 'classify_file'):
                        raw = _audio_classifier.classify_file(in_path)
                    else:
                        # 尝试通用接口
                        raw = _audio_classifier.inference(in_path)
                except Exception as e:
                    logger.debug(f"SpeechBrain predict attempt failed: {e}")

                timeline = []
                if isinstance(raw, str):
                    timeline = [{'start': 0.0, 'end': 0.0, 'emotion': [{'label': raw, 'score': 1.0}]}]
                elif isinstance(raw, dict):
                    if 'label' in raw and 'score' in raw:
                        timeline = [{'start': 0.0, 'end': 0.0, 'emotion': [{'label': raw.get('label'), 'score': float(raw.get('score', 1.0))}]}]
                    else:
                        labels = []
                        for k, v in raw.items():
                            try:
                                labels.append({'label': str(k), 'score': float(v)})
                            except Exception:
                                pass
                        if labels:
                            timeline = [{'start': 0.0, 'end': 0.0, 'emotion': labels}]
                elif isinstance(raw, (list, tuple)) and raw:
                    if all(isinstance(item, dict) for item in raw):
                        labels = []
                        for item in raw:
                            if 'label' in item:
                                labels.append({'label': item.get('label'), 'score': float(item.get('score', 1.0))})
                        if labels:
                            timeline = [{'start': 0.0, 'end': 0.0, 'emotion': labels}]

                best_label = 'neutral'
                best_score = 0.0
                for seg in timeline:
                    for e in seg.get('emotion', []):
                        if e.get('score', 0) > best_score:
                            best_score = e.get('score', 0)
                            best_label = e.get('label') or best_label

                result = {'timeline': timeline, 'summary': best_label}
                # 空结果时优先尝试备用 HF 声学模型，失败再考虑文本回退
                if not result['timeline']:
                    backup_id = os.environ.get('AUDIO_BACKUP_HF_ID', 'superb/hubert-large-superb-er')
                    try:
                        from transformers import pipeline as hf_pipeline
                        try:
                            import torch
                            device2 = 0 if torch.cuda.is_available() else -1
                        except Exception:
                            device2 = -1
                        backup_clf = hf_pipeline('audio-classification', model=backup_id, device=device2)
                        raw2 = None
                        # 先文件调用
                        try:
                            raw2 = backup_clf(in_path)
                        except Exception:
                            # 退回数组调用
                            try:
                                import librosa
                                arr, sr = librosa.load(in_path, sr=None, mono=True)
                                raw2 = backup_clf({'array': arr, 'sampling_rate': int(sr)})
                            except Exception:
                                raw2 = None
                        if isinstance(raw2, list) and raw2:
                            labels = []
                            for r2 in raw2:
                                if isinstance(r2, dict) and 'label' in r2:
                                    labels.append({'label': r2.get('label'), 'score': float(r2.get('score', 0))})
                            if labels:
                                result['timeline'] = [{'start': 0.0, 'end': 0.0, 'emotion': labels}]
                                # 同步 summary
                                if labels:
                                    m = max(labels, key=lambda x: x.get('score', 0))
                                    result['summary'] = m.get('label', result['summary'])
                                if os.environ.get('AUDIO_DEBUG') == '1':
                                    result.setdefault('_debug', {})
                                    result['_debug']['backup_model_id'] = backup_id
                                    result['_debug']['backup_backend'] = 'transformers'
                    except Exception as be:
                        # 仅记录，不中断
                        if os.environ.get('AUDIO_DEBUG') == '1':
                            result.setdefault('_debug', {})
                            result['_debug']['backup_error'] = str(be)

                    # 若仍为空，且允许，则文本回退
                    if not result['timeline'] and os.environ.get('AUDIO_ALLOW_TEXT_FALLBACK', '1') == '1':
                        try:
                            from . import text_service
                            text_fb = text_service.predict(audio_path)
                            result['timeline'] = _text_result_to_audio_timeline(text_fb)
                            if os.environ.get('AUDIO_DEBUG') == '1':
                                result.setdefault('_debug', {})
                                result['_debug']['fallback'] = 'speechbrain_empty_to_asr_text'
                                result['_debug']['text_result_sample'] = text_fb.get('timeline', [])[:1]
                        except Exception as fb_e:
                            if os.environ.get('AUDIO_DEBUG') == '1':
                                result.setdefault('_debug', {})
                                result['_debug']['fallback_attempt_error'] = str(fb_e)

                if os.environ.get('AUDIO_DEBUG') == '1':
                    dbg = result.get('_debug')
                    if not isinstance(dbg, dict):
                        dbg = {}

                    effective_backend = 'speechbrain'
                    # 如果使用了备用 HF 模型拿到了结果，标记实际 backend
                    if dbg.get('backup_backend') == 'transformers' and result.get('timeline'):
                        effective_backend = 'transformers_backup'
                    # 如果最终回退到了文本情感，则标记为 asr_text
                    if isinstance(dbg.get('fallback'), str) and 'asr_text' in dbg.get('fallback', ''):
                        effective_backend = 'asr_text'

                    if not dbg.get('backend'):
                        dbg['backend'] = effective_backend
                    if not dbg.get('audio_model_id'):
                        dbg['audio_model_id'] = _audio_model_id
                    if 'audio_load_error' not in dbg or dbg.get('audio_load_error') is None:
                        dbg['audio_load_error'] = _audio_load_error
                    if 'raw_info' not in dbg or dbg.get('raw_info') is None:
                        dbg['raw_info'] = {'type': str(type(raw)), 'repr': str(raw)[:1000]}
                    result['_debug'] = dbg
                return result
            except Exception as e:
                logger.warning(f"SpeechBrain prediction failed: {e}")
                if os.environ.get('AUDIO_DEBUG') == '1':
                    return {'timeline': [], 'summary': 'neutral', '_debug': {'reason': 'speechbrain_failed', 'error': str(e), 'audio_load_error': _audio_load_error}}
                return {'timeline': [], 'summary': 'neutral'}

        raw = None
        debug_calls = {'model_id': _audio_model_id, 'device': _audio_device, 'tried_file_call': False, 'tried_array_call': False, 'array_loader': None, 'sampling_rate': None, 'array_len': None, 'exceptions': []}

        # 1) 先用文件路径调用 pipeline
        try:
            debug_calls['tried_file_call'] = True
            raw = classifier(in_path)
        except Exception as e:
            logger.debug(f"audio classifier call with file path failed: {e}")
            debug_calls['exceptions'].append({'file_call': str(e)})

        # 2) 如果返回空或调用失败，尝试以数组形式调用 pipeline（需要 torchaudio 或 librosa）
        if not raw:
            debug_calls['tried_array_call'] = True
            # 尝试加载为 numpy array，优先 torchaudio
            try:
                try:
                    import torchaudio
                    waveform, sr = torchaudio.load(in_path)
                    # waveform shape: (channels, frames)
                    if hasattr(waveform, 'numpy'):
                        import numpy as np
                        arr = waveform.mean(axis=0).numpy()
                    else:
                        arr = waveform.mean(axis=0)
                    debug_calls['array_loader'] = 'torchaudio'
                    debug_calls['sampling_rate'] = int(sr)
                    debug_calls['array_len'] = int(len(arr))
                    raw = classifier({'array': arr, 'sampling_rate': int(sr)})
                except Exception as e1:
                    debug_calls['exceptions'].append({'torchaudio': str(e1)})
                    # fallback to librosa
                    try:
                        import librosa
                        arr, sr = librosa.load(in_path, sr=None, mono=True)
                        debug_calls['array_loader'] = 'librosa'
                        debug_calls['sampling_rate'] = int(sr)
                        debug_calls['array_len'] = int(len(arr))
                        raw = classifier({'array': arr, 'sampling_rate': int(sr)})
                    except Exception as e2:
                        debug_calls['exceptions'].append({'librosa': str(e2)})
                        # final fallback: stdlib wave (no soundfile dependency)
                        import wave
                        import numpy as np
                        with wave.open(in_path, 'rb') as wf:
                            sr = wf.getframerate()
                            n_channels = wf.getnchannels()
                            sampwidth = wf.getsampwidth()
                            n_frames = wf.getnframes()
                            frames = wf.readframes(n_frames)
                        if sampwidth != 2:
                            raise RuntimeError(f"Unsupported WAV sampwidth={sampwidth}")
                        x = np.frombuffer(frames, dtype=np.int16)
                        if n_channels > 1:
                            x = x.reshape(-1, n_channels).mean(axis=1)
                        arr = (x.astype(np.float32) / 32768.0)
                        # resample to 16k if possible (pure torch, no backend)
                        try:
                            import torch
                            import torchaudio
                            if int(sr) != 16000:
                                t = torch.from_numpy(arr)
                                t = torchaudio.functional.resample(t, int(sr), 16000)
                                arr = t.numpy()
                                sr = 16000
                        except Exception as e3:
                            debug_calls['exceptions'].append({'wave_resample': str(e3)})
                        debug_calls['array_loader'] = 'wave'
                        debug_calls['sampling_rate'] = int(sr)
                        debug_calls['array_len'] = int(len(arr))
                        raw = classifier({'array': arr, 'sampling_rate': int(sr)})
            except Exception as e:
                logger.debug(f"audio classifier call with array failed: {e}")
                debug_calls['exceptions'].append({'array_call': str(e)})

        # raw 可能是一个 list，元素包含 'label','score'，或者包含 segments
        timeline = []
        # possible formats:
        # - list of {'label','score'} -> single-segment multi-label
        # - list of segments with 'start'/'end' and either 'label'/'score' or 'emotion'
        # - dict with 'labels' or 'chunks'
        if isinstance(raw, list) and raw:
            # 检查是否是 segment 列表
            if all(('start' in r and 'end' in r) for r in raw):
                for r in raw:
                    # segment may have 'label' and 'score' or 'emotion' list
                    if 'emotion' in r and isinstance(r['emotion'], list):
                        labels = [{'label': e.get('label'), 'score': float(e.get('score', 0))} for e in r['emotion']]
                    else:
                        labels = [{'label': r.get('label'), 'score': float(r.get('score', 0))}]
                    timeline.append({'start': float(r.get('start', 0)), 'end': float(r.get('end', 0)), 'emotion': labels})
            else:
                # treat as single-segment multi-label
                labels = []
                for r in raw:
                    if isinstance(r, dict) and 'label' in r:
                        labels.append({'label': r.get('label'), 'score': float(r.get('score', 0))})
                if labels:
                    timeline = [{'start': 0.0, 'end': 0.0, 'emotion': labels}]
        elif isinstance(raw, dict):
            # common keys: 'labels', 'chunks', 'segments'
            if 'labels' in raw and isinstance(raw['labels'], list):
                labels = [{'label': r.get('label') if isinstance(r, dict) else str(r), 'score': float(r.get('score', 0) if isinstance(r, dict) else 0)} for r in raw['labels']]
                timeline = [{'start': 0.0, 'end': 0.0, 'emotion': labels}]
            elif 'chunks' in raw and isinstance(raw['chunks'], list):
                for c in raw['chunks']:
                    # chunk might have 'start','end','text' or 'emotion'
                    if 'emotion' in c and isinstance(c['emotion'], list):
                        labels = [{'label': e.get('label'), 'score': float(e.get('score', 0))} for e in c['emotion']]
                    elif 'label' in c:
                        labels = [{'label': c.get('label'), 'score': float(c.get('score', 0))}]
                    else:
                        labels = []
                    timeline.append({'start': float(c.get('start', 0)), 'end': float(c.get('end', 0)), 'emotion': labels})
        else:
            logger.debug(f"audio classifier returned unsupported format: {type(raw)}")

        # compute summary: pick highest-scoring label across timeline
        best_label = 'neutral'
        best_score = 0.0
        for seg in timeline:
            for e in seg.get('emotion', []):
                if e.get('score', 0) > best_score:
                    best_score = e.get('score', 0)
                    best_label = e.get('label') or best_label

        result = {'timeline': timeline, 'summary': best_label}

        # 如果 acoustic classifier 返回空 timeline，作为最后保险回退到文本情感（ASR->text）
        if not result['timeline']:
            try:
                # 可通过环境变量关闭此回退：AUDIO_ALLOW_TEXT_FALLBACK=0
                if os.environ.get('AUDIO_ALLOW_TEXT_FALLBACK', '1') == '1':
                    from . import text_service
                    text_fallback = text_service.predict(audio_path)
                    result['timeline'] = _text_result_to_audio_timeline(text_fallback)
                    # 标记来源以便上层知道这是回退结果
                    if os.environ.get('AUDIO_DEBUG') == '1':
                        result.setdefault('_debug', {})
                        result['_debug']['fallback'] = 'asr_text_post'
                        result['_debug']['audio_model_id'] = _audio_model_id
                        result['_debug']['audio_load_error'] = _audio_load_error
                        result['_debug']['text_result_sample'] = text_fallback.get('timeline', [])[:1]
                else:
                    if os.environ.get('AUDIO_DEBUG') == '1':
                        result.setdefault('_debug', {})
                        result['_debug']['fallback'] = 'disabled'
            except Exception as e:
                # 保持原样（空 timeline）并记录 debug
                if os.environ.get('AUDIO_DEBUG') == '1':
                    result.setdefault('_debug', {})
                    result['_debug']['fallback_attempt_error'] = str(e)

        if os.environ.get('AUDIO_DEBUG') == '1':
            try:
                raw_info = None
                if raw is None:
                    raw_info = None
                elif isinstance(raw, list):
                    # include a tiny sample if small
                    sample = raw[:3] if len(raw) <= 3 else [ {k:v for k,v in r.items() if k in ('label','score','start','end') } for r in raw[:3] ]
                    raw_info = {'type': 'list', 'len': len(raw), 'sample': sample}
                elif isinstance(raw, dict):
                    keys = list(raw.keys())
                    raw_info = {'type': 'dict', 'keys': keys}
                else:
                    raw_info = {'type': str(type(raw))}

                dbg = result.get('_debug')
                if not isinstance(dbg, dict):
                    dbg = {}
                effective_backend = _audio_backend or 'transformers'
                if isinstance(dbg.get('fallback'), str) and 'asr_text' in dbg.get('fallback', ''):
                    effective_backend = 'asr_text'
                if not dbg.get('backend'):
                    dbg['backend'] = effective_backend
                if not dbg.get('audio_model_id'):
                    dbg['audio_model_id'] = _audio_model_id
                if 'audio_load_error' not in dbg or dbg.get('audio_load_error') is None:
                    dbg['audio_load_error'] = _audio_load_error
                dbg['raw_info'] = raw_info
                dbg['calls'] = debug_calls
                result['_debug'] = dbg
            except Exception:
                dbg = result.get('_debug')
                if not isinstance(dbg, dict):
                    dbg = {}
                effective_backend = _audio_backend or 'transformers'
                if isinstance(dbg.get('fallback'), str) and 'asr_text' in dbg.get('fallback', ''):
                    effective_backend = 'asr_text'
                if not dbg.get('backend'):
                    dbg['backend'] = effective_backend
                if not dbg.get('audio_model_id'):
                    dbg['audio_model_id'] = _audio_model_id
                if 'audio_load_error' not in dbg or dbg.get('audio_load_error') is None:
                    dbg['audio_load_error'] = _audio_load_error
                dbg['raw_info'] = 'unavailable'
                dbg['calls'] = debug_calls
                result['_debug'] = dbg

        return result

    except Exception as e:
        logger.warning(f"Audio prediction failed: {e}")
        return {'timeline': [], 'summary': 'neutral'}
    finally:
        # 清理临时文件
        try:
            if tmp_path and tmp_path != audio_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
