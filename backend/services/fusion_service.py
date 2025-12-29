# 学习式多模态融合：将各模态 logits 与时长/置信度特征拼接，经 2 层 MLP（hidden=64）输出最终分类
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

# 7 类固定顺序，便于向量化
LABEL_ORDER: List[str] = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# MLP 输入维度：3 个模态的 logits（7*3）+ 各模态时长与置信度（3+3）
_INPUT_DIM = len(LABEL_ORDER) * 3 + 6
_HIDDEN_DIM = 64

_mlp_params: Dict[str, List[List[float]]] = {}


def _normalize_label(label: str) -> str:
    if not label:
        return 'neutral'
    l = str(label).strip().lower()
    if l == 'anger':
        return 'angry'
    if l == 'sadness':
        return 'sad'
    if l == 'joy':
        return 'happy'
    if l == 'calm':
        return 'neutral'
    return l


def _aggregate_timeline_scores(timeline) -> Dict[str, float]:
    scores: Dict[str, float] = defaultdict(float)
    if not timeline:
        return scores
    for seg in timeline:
        emos = seg.get('emotion') or []
        for e in emos:
            lbl = _normalize_label(e.get('label'))
            sc = float(e.get('score', 0) or 0)
            scores[lbl] += sc
    return scores


def _duration_and_confidence(timeline) -> Tuple[float, float]:
    duration = 0.0
    max_conf = 0.0
    if not timeline:
        return duration, max_conf
    for seg in timeline:
        try:
            start = float(seg.get('start', 0) or 0)
            end = float(seg.get('end', 0) or 0)
            if end > start:
                duration += (end - start)
        except Exception:
            pass
        emos = seg.get('emotion') or []
        for e in emos:
            try:
                sc = float(e.get('score', 0) or 0)
                if sc > max_conf:
                    max_conf = sc
            except Exception:
                continue
    return duration, max_conf


def _vectorize_scores(score_map: Dict[str, float], summary_label: str = None) -> List[float]:
    vec = [0.0 for _ in LABEL_ORDER]
    for idx, lbl in enumerate(LABEL_ORDER):
        if lbl in score_map:
            vec[idx] = float(score_map[lbl])
    if not any(vec) and summary_label:
        try:
            vec[LABEL_ORDER.index(_normalize_label(summary_label))] = 1.0
        except ValueError:
            pass
    return vec


def _extract_features(mod_result: dict) -> Tuple[List[float], float, float]:
    """返回 (logits_vec, duration, confidence) 三元组。"""
    if not isinstance(mod_result, dict):
        return [0.0] * len(LABEL_ORDER), 0.0, 0.0
    timeline = mod_result.get('timeline', []) if isinstance(mod_result, dict) else []
    scores = _aggregate_timeline_scores(timeline)
    duration, conf = _duration_and_confidence(timeline)
    vec = _vectorize_scores(scores, mod_result.get('summary'))
    return vec, duration, conf


def _init_mlp_params():
    # 允许通过环境变量 FUSION_MLP_WEIGHTS 指定 JSON 权重文件
    global _mlp_params
    if _mlp_params:
        return

    weight_path = os.environ.get('FUSION_MLP_WEIGHTS')
    if weight_path and os.path.isfile(weight_path):
        try:
            with open(weight_path, 'r', encoding='utf-8') as f:
                params = json.load(f)
            w1 = params.get('w1', [])
            b1 = params.get('b1', [])
            w2 = params.get('w2', [])
            b2 = params.get('b2', [])
            if len(w1) == _HIDDEN_DIM and len(w1[0]) == _INPUT_DIM and len(w2) == len(LABEL_ORDER) and len(w2[0]) == _HIDDEN_DIM:
                if len(b1) == _HIDDEN_DIM and len(b2) == len(LABEL_ORDER):
                    _mlp_params = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}
                    return
        except Exception:
            # 读取失败则回退到默认权重
            pass

    # 构造可解释的确定性默认权重：前 7 个隐藏单元汇总三模态同类 logits，并轻度加权时长/置信度
    w1 = [[0.0 for _ in range(_INPUT_DIM)] for _ in range(_HIDDEN_DIM)]
    b1 = [0.0 for _ in range(_HIDDEN_DIM)]
    for i in range(len(LABEL_ORDER)):
        # text/audio/image logits 加总
        w1[i][i] = 1.0
        w1[i][len(LABEL_ORDER) + i] = 1.0
        w1[i][len(LABEL_ORDER) * 2 + i] = 1.0
        # 时长和置信度给予微小增益（防止全零时仍可被 neutral 偏置接管）
        dur_base = len(LABEL_ORDER) * 3
        conf_base = dur_base + 3
        w1[i][dur_base + (i % 3)] = 0.05
        w1[i][conf_base + (i % 3)] = 0.1

    w2 = [[0.0 for _ in range(_HIDDEN_DIM)] for _ in range(len(LABEL_ORDER))]
    b2 = [-0.05 for _ in range(len(LABEL_ORDER))]
    neutral_idx = LABEL_ORDER.index('neutral')
    b2[neutral_idx] = 0.05
    for i in range(len(LABEL_ORDER)):
        w2[i][i] = 1.0

    _mlp_params = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}


def _relu(vec: List[float]) -> List[float]:
    return [v if v > 0 else 0.0 for v in vec]


def _dense(x: List[float], w: List[List[float]], b: List[float]) -> List[float]:
    out = []
    for row, bias in zip(w, b):
        s = bias
        for xv, rv in zip(x, row):
            s += xv * rv
        out.append(s)
    return out


def _mlp_forward(x: List[float]) -> List[float]:
    _init_mlp_params()
    h = _dense(x, _mlp_params['w1'], _mlp_params['b1'])
    h = _relu(h)
    o = _dense(h, _mlp_params['w2'], _mlp_params['b2'])
    return o


def fuse(text_result: dict, audio_result: dict, image_result: dict, weights: dict = None) -> dict:
    """学习式融合：
    - 输入：三模态的 logits（时间轴情感分数视作 logits）、时长、置信度
    - 模型：2 层 MLP（hidden=64，ReLU），默认权重可解释，亦可通过 FUSION_MLP_WEIGHTS 指定 JSON 覆盖
    - 输出：{'summary': label, 'logits': {label: score}, 'timeline': {...}}
    """
    # 抽取模态特征
    text_logits, text_dur, text_conf = _extract_features(text_result)
    audio_logits, audio_dur, audio_conf = _extract_features(audio_result)
    image_logits, image_dur, image_conf = _extract_features(image_result)

    features: List[float] = []
    features.extend(text_logits)
    features.extend(audio_logits)
    features.extend(image_logits)
    features.extend([text_dur, audio_dur, image_dur, text_conf, audio_conf, image_conf])

    outputs = _mlp_forward(features)
    best_idx = max(range(len(outputs)), key=lambda i: outputs[i]) if outputs else LABEL_ORDER.index('neutral')
    summary = LABEL_ORDER[best_idx]
    logits_map = {lbl: float(outputs[i]) for i, lbl in enumerate(LABEL_ORDER)}

    timeline = {
        'text': text_result.get('timeline', []) if isinstance(text_result, dict) else [],
        'audio': audio_result.get('timeline', []) if isinstance(audio_result, dict) else [],
        'image': image_result.get('timeline', []) if isinstance(image_result, dict) else []
    }

    return {
        'timeline': timeline,
        'summary': summary,
        'logits': logits_map,
        'features_used': {
            'text': {'duration': text_dur, 'confidence': text_conf},
            'audio': {'duration': audio_dur, 'confidence': audio_conf},
            'image': {'duration': image_dur, 'confidence': image_conf},
        },
    }
