# app_zh.py
# -*- coding: utf-8 -*-
"""
情感支持机器人（中文版，表情增强）
--------------------------------
依赖:
    pip install "flask>=2.2" "transformers>=4.40" "torch>=2.2" emoji

启动:
    python app_zh.py
"""

import os
import re
import sys
import time
import random
import errno

import torch
from flask import Flask, request, jsonify
from transformers import pipeline

# ============================= Windows 控制台编码修复 =============================
if sys.platform == "win32":
    os.system("")  # 启用 VT100 转义序列支持
    try:
        if sys.stdout.encoding != "utf-8":
            sys.stdout = open(sys.stdout.fileno(), "w", encoding="utf-8", errors="ignore")
        if sys.stderr.encoding != "utf-8":
            sys.stderr = open(sys.stderr.fileno(), "w", encoding="utf-8", errors="ignore")
    except Exception:
        pass


def safe_print(*args, **kwargs):
    """跨平台安全打印（含 Windows 控制台）"""
    try:
        message = " ".join(str(arg) for arg in args)
        if sys.platform == "win32":
            try:
                print(message.encode("utf-8", "ignore").decode("utf-8", "ignore"), **kwargs)
            except Exception:
                encoding = sys.stdout.encoding or "utf-8"
                print(message.encode(encoding, "ignore").decode(encoding, "ignore"), **kwargs)
        else:
            print(message, **kwargs)
    except Exception as e:
        print(f"[PRINT ERROR] {e}", file=sys.stderr)


# ============================= 运行环境设置 =============================
# 如不需要镜像可注释
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False  # JSON 保持中文与表情

# 轻量安全响应头
@app.after_request
def add_security_headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Cache-Control"] = "no-store"
    return resp


# ============================= 业务逻辑 =============================
try:
    import emoji as emoji_lib  # pip install emoji
except Exception:
    emoji_lib = None


class EmotionalSupportBotCN:
    def __init__(self):
        safe_print("😊 正在初始化中文情绪识别（零样本分类）...")

        device = 0 if torch.cuda.is_available() else -1

        # 中文零样本情绪分类（支持多语）
        # 候选标签：愤怒/快乐/悲伤/恐惧/爱/惊讶/中性
        self.labels_cn = ["悲伤", "快乐", "愤怒", "恐惧", "爱", "惊讶", "中性"]
        self.hypothesis_template = "这段话表达了{}。"
        self.emotion_classifier = pipeline(
            "zero-shot-classification",
            model="joeddav/xlm-roberta-large-xnli",
            device=device,
        )

        safe_print("💬 正在初始化中文生成模型...")
        # 轻量中文 T5
        self.response_generator = pipeline(
            "text2text-generation",
            model="uer/t5-small-chinese-cluecorpussmall",
            device=device,
        )

        # 预设中文共情回应
        self.empathy_responses = {
            "悲伤": [
                "😢 我听见你在难过，这种感受真的很不容易。",
                "🤗 想聊聊发生了什么吗？我很在意你的感受。",
                "💔 难过是会来的，也会过去。你并不孤单。",
                "🌧️ 允许自己难过一下没关系，我会在这里陪你。",
            ],
            "快乐": [
                "🎉 太好了！能分享一下让你开心的事吗？",
                "😄 你的喜悦让我也感到温暖！",
                "🌈 这份快乐很珍贵，想多说一点吗？",
                "☀️ 看到你开心真好！",
            ],
            "愤怒": [
                "😠 我能感到你的愤怒，这是合理的感受。",
                "💢 想说说让你生气的根源吗？我愿意倾听。",
                "🧘 先深呼吸一下，我们慢慢梳理发生了什么。",
                "⚡ 生气时很难想清楚，我可以陪你理一理。",
            ],
            "恐惧": [
                "😨 我理解你在担心，害怕是很自然的反应。",
                "🛡️ 有时恐惧只是想保护我们，我们可以一步步来。",
                "🤝 不用一个人面对，我会陪你一起想办法。",
            ],
            "爱": [
                "💖 这是很美好的情感。愿意多分享一点吗？",
                "❤️ 被爱与去爱都很珍贵。",
                "💕 这份在乎会让世界不一样。",
            ],
            "惊讶": [
                "😲 哇，真让人意外！发生了什么？",
                "✨ 意外有时会带来新的视角，想聊聊吗？",
                "🎯 或许这是个转折点，你怎么看？",
            ],
            "中性": [
                "😌 我在这儿，想多说一点也可以。",
                "💬 你还想聊些什么？",
                "🤔 挺有意思的，能再具体一点吗？",
            ],
        }

        # 表情与中文情绪映射（含❤️/❤）
        self.emoji_emotion_map = {
            "😠": "愤怒",
            "😡": "愤怒",
            "💢": "愤怒",
            "😤": "愤怒",
            "🤬": "愤怒",
            "😃": "快乐",
            "😄": "快乐",
            "😁": "快乐",
            "🥳": "快乐",
            "🤩": "快乐",
            "😂": "快乐",
            "😅": "快乐",
            "😇": "快乐",
            "🤣": "快乐",
            "🙂": "快乐",
            "😉": "快乐",
            "😊": "快乐",
            "🥰": "爱",
            "😘": "爱",
            "😍": "爱",
            "❤️": "爱",
            "❤": "爱",
            "💕": "爱",
            "💘": "爱",
            "💖": "爱",
            "💗": "爱",
            "💓": "爱",
            "💞": "爱",
            "🤗": "爱",
            "😢": "悲伤",
            "😭": "悲伤",
            "😿": "悲伤",
            "😓": "悲伤",
            "😞": "悲伤",
            "😔": "悲伤",
            "🥺": "悲伤",
            "😥": "悲伤",
            "😰": "恐惧",
            "😨": "恐惧",
            "😧": "恐惧",
            "😬": "恐惧",
            "😱": "恐惧",
            "👻": "恐惧",
            "😲": "惊讶",
            "😯": "惊讶",
            "🤯": "惊讶",
            "🤔": "中性",
            "😐": "中性",
            "😑": "中性",
            "🙄": "中性",
            "🧐": "中性",
        }

        self.emoji_description = {
            "😠": "生气的脸",
            "😡": "气鼓鼓的脸",
            "😃": "笑脸",
            "😄": "露齿笑",
            "😁": "咧嘴笑",
            "🥳": "庆祝脸",
            "🤩": "星星眼",
            "😂": "笑哭",
            "😅": "尴尬笑",
            "😇": "天使笑脸",
            "🤣": "笑到打滚",
            "🙂": "微笑",
            "😉": "眨眼",
            "😊": "暖笑",
            "🥰": "爱心满满",
            "😘": "飞吻",
            "😍": "爱心眼",
            "❤️": "红心",
            "❤": "红心",
            "💕": "双心",
            "💘": "丘比特之心",
            "💖": "亮晶晶的心",
            "💗": "渐变心",
            "💓": "跳动的心",
            "💞": "旋转的心",
            "🤗": "拥抱",
            "😢": "流泪",
            "😭": "大哭",
            "😿": "哭泣猫",
            "😓": "汗颜",
            "😞": "失望",
            "😔": "沉思",
            "🥺": "恳求",
            "😥": "如释重负",
            "😰": "焦虑",
            "😨": "害怕",
            "😧": "痛苦",
            "😬": "龇牙",
            "😱": "惊恐尖叫",
            "👻": "幽灵",
            "😲": "惊讶",
            "😯": "噤声",
            "🤯": "脑洞大开",
            "🤔": "思考",
            "😐": "中性",
            "😑": "无表情",
            "🙄": "翻白眼",
            "🧐": "单片眼镜",
        }

        safe_print("🤖 中文情感支持机器人初始化完成")

    def detect_emotion(self, user_input: str):
        """返回 (情绪中文标签, 检出的表情描述串)"""
        if not user_input.strip():
            return None, ""

        # 兼容多码点表情（如 ❤️）
        if emoji_lib is not None:
            emojis = [e["emoji"] for e in emoji_lib.emoji_list(user_input)]
        else:
            keys = sorted(self.emoji_emotion_map.keys(), key=len, reverse=True)
            emojis, i = [], 0
            while i < len(user_input):
                for k in keys:
                    if user_input.startswith(k, i):
                        emojis.append(k)
                        i += len(k)
                        break
                else:
                    i += 1

        emoji_descs = [self.emoji_description.get(c, "表情") for c in emojis]
        detected_emojis = ""

        try:
            # 零样本中文情绪分类
            z = self.emotion_classifier(
                user_input,
                candidate_labels=self.labels_cn,
                hypothesis_template=self.hypothesis_template,
                multi_label=False,
            )
            base_emotion = z["labels"][0]
            base_score = z["scores"][0]

            # 合并表情线索
            if emojis:
                from collections import Counter

                cnt = Counter(self.emoji_emotion_map.get(e) for e in emojis if e in self.emoji_emotion_map)
                if cnt:
                    emoji_emotion, freq = cnt.most_common(1)[0]
                    emoji_conf = freq / max(1, len(emojis))
                    final_emotion = emoji_emotion if (emoji_conf > 0.5 and base_score < 0.70) else base_emotion
                else:
                    final_emotion = base_emotion
            else:
                final_emotion = base_emotion

            detected_emojis = "、".join(emoji_descs) if emoji_descs else "无"
            safe_print(f"🎭 检测到情绪: {final_emotion} (文本:{base_emotion:.2f}, 表情:{detected_emojis})")
            return final_emotion, detected_emojis
        except Exception as e:
            safe_print(f"❗ 情绪分析出错: {e}")
            detected_emojis = "、".join(emoji_descs) if emoji_descs else "无"
            return "中性", detected_emojis

    def generate_response(self, user_input: str, emotion_cn: str, detected_emojis: str):
        # 优先使用预设中文共情回复
        if emotion_cn in self.empathy_responses:
            resp = random.choice(self.empathy_responses[emotion_cn])
            safe_print(f"💡 使用预设中文回应: {resp}")
            return resp

        # 中文提示词
        emoji_ctx = f"（注意到你使用了这些表情：{detected_emojis}）" if detected_emojis != "无" else ""
        prompt = f"请用温暖、理解、支持的语气，用简洁中文回答：{emoji_ctx} 用户说：{user_input}"
        safe_print(f"💭 生成提示: {prompt}")

        try:
            out = self.response_generator(
                prompt,
                max_new_tokens=80,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                num_return_sequences=1,
            )
            resp = out[0]["generated_text"]

            # 若缺少表情，根据情绪补一个
            if not any(ch in resp for ch in "😀😁😂😃😄😅😉😊😍😘😇😐😑😶😏😣😥😮😢😨😠❤️"):
                emo_emoji = {
                    "愤怒": "😠",
                    "快乐": "😄",
                    "悲伤": "😢",
                    "恐惧": "😨",
                    "爱": "❤️",
                    "惊讶": "😲",
                    "中性": "🤔",
                }.get(emotion_cn, "🤖")
                resp = f"{emo_emoji} {resp}"

            return resp
        except Exception as e:
            safe_print(f"❗ 生成回应出错: {e}")
            return "😕 我能感受到你的情绪，但还需要你再多说一点，我会认真听你讲。"

    def validate_response(self, response: str, user_input: str):
        low = response.lower()
        if any(w in low for w in ["不适当", "负面", "伤害", "inappropriate", "harmful"]):
            return "🙏 抱歉，我不确定怎样更合适地回应。我们可以换个角度慢慢聊聊吗？"
        if "不理解" in response or "不知道" in response or "don't understand" in low:
            return f"📚 我还在学习理解情绪。你刚刚说「{user_input}」，可以再具体一些吗？"
        return response


safe_print("🤖 启动中文情感支持机器人...")
bot = EmotionalSupportBotCN()


# ============================= 路由 =============================
@app.route("/chat", methods=["POST"])
def chat():
    safe_print("\n📩 收到聊天请求...")
    try:
        data = request.get_json(silent=True) or {}
        user_input = (data.get("message") or "").strip()
        safe_print(f"🗣️ 用户输入: {user_input}")

        if not user_input:
            return jsonify(
                {
                    "response": "😶 我这边没听清，你可以再说一次吗？",
                    "emotion": "中性",
                    "emojis": "",
                    "end_conversation": False,
                }
            )

        if re.search(r"\b(bye|goodbye|exit|quit|再见|拜拜|退出)\b", user_input, re.I):
            safe_print("👋 结束对话请求")
            return jsonify(
                {
                    "response": "💖 谢谢你的分享。记住你并不孤单，需要我时我都会在。",
                    "end_conversation": True,
                }
            )

        emotion, detected_emojis = bot.detect_emotion(user_input)
        response = bot.generate_response(user_input, emotion, detected_emojis)
        final_response = bot.validate_response(response, user_input)

        safe_print(f"💬 发送回应: {final_response}")
        return jsonify(
            {
                "response": final_response,
                "emotion": emotion,
                "emojis": detected_emojis,
                "end_conversation": False,
            }
        )
    except Exception as e:
        safe_print(f"❗ 聊天处理错误: {e}")
        return jsonify(
            {
                "response": "⚠️ 我这边遇到点小问题，可以稍后再试一次吗？",
                "emotion": "中性",
                "emojis": "",
                "end_conversation": False,
            }
        )


@app.route("/")
def index():
    """中文前端（已修复 XSS，用 textContent 渲染用户/机器人文本）"""
    try:
        html = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>情感支持机器人 - 小Y</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
*{box-sizing:border-box;margin:0;padding:0;font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif}
body{
  background:linear-gradient(135deg,#1e88e5,#1976d2);
  display:flex;justify-content:center;align-items:center;min-height:100vh;padding:20px;color:#333
}
.chat-container{
  width:100%;max-width:500px;height:90vh;background-color:rgba(255,255,255,.95);
  border-radius:20px;box-shadow:0 10px 30px rgba(0,0,0,.3);display:flex;flex-direction:column;overflow:hidden;
  border:1px solid rgba(255,255,255,.2)
}
.chat-header{
  background:linear-gradient(135deg,#1e88e5,#1976d2);color:#fff;padding:20px;text-align:center;font-size:1.2rem;font-weight:600;position:relative;
  box-shadow:0 4px 12px rgba(0,0,0,.2);border-bottom:2px solid rgba(255,255,255,.2)
}
.header-icon{font-size:1.6rem;margin-right:10px;vertical-align:middle}
.chat-messages{flex:1;padding:20px;overflow-y:auto;background:rgba(235,245,255,.7);display:flex;flex-direction:column}
.welcome-message{
  text-align:center;margin-bottom:20px;color:#0d47a1;font-size:.95rem;line-height:1.6;background:rgba(255,255,255,.9);padding:15px;border-radius:15px;
  box-shadow:0 4px 15px rgba(0,0,0,.1);border:1px solid rgba(25,118,210,.2)
}
.message{
  max-width:85%;padding:15px 20px;margin-bottom:15px;border-radius:20px;line-height:1.5;position:relative;animation:fadeIn .4s ease;font-size:1.05rem;
  box-shadow:0 4px 8px rgba(0,0,0,.1);transition:transform .3s,box-shadow .3s;overflow-wrap:break-word;border:1px solid rgba(0,0,0,.05)
}
.user-message{
  background:linear-gradient(135deg,#1e88e5,#1976d2);color:#fff;margin-left:auto;border-bottom-right-radius:5px;
  box-shadow:0 4px 10px rgba(30,136,229,.3);transform-origin:right;text-align:right
}
.bot-message{
  background:linear-gradient(135deg,#fff,#f8f9fa);color:#333;margin-right:auto;border-bottom-left-radius:5px;
  box-shadow:0 4px 15px rgba(0,0,0,.08);transform-origin:left;text-align:left
}
.message:hover{transform:translateY(-3px);box-shadow:0 6px 15px rgba(0,0,0,.15)}
.typing-indicator{
  display:none;padding:15px 20px;background:#f8f9fa;color:#333;border-radius:20px;margin-bottom:15px;width:fit-content;border-bottom-left-radius:5px;
  box-shadow:0 4px 15px rgba(0,0,0,.08);font-size:1.05rem
}
.emotion-tag{
  font-size:.85rem;color:#1976d2;margin-top:8px;font-weight:500;text-align:left;display:flex;align-items:center;padding:8px 15px;background:rgba(30,136,229,.1);
  border-radius:15px;margin-left:auto;margin-right:auto;width:fit-content;box-shadow:0 2px 5px rgba(0,0,0,.05)
}
.chat-input-container{display:flex;padding:15px 20px;background:#fff;border-top:1px solid rgba(0,0,0,.1)}
.chat-input-group{flex:1;display:flex;position:relative}
.chat-tools{display:flex;align-items:center;margin-bottom:10px;margin-left:10px;background:rgba(30,136,229,.05);padding:8px 15px;border-radius:25px}
.quick-emoji-btn{font-size:1.4rem;background:none;border:none;cursor:pointer;margin:0 5px;padding:8px;border-radius:50%;transition:all .3s}
.quick-emoji-btn:hover{background:rgba(30,136,229,.15);transform:scale(1.15)}
#message-input{
  flex:1;padding:15px 20px;border:2px solid #bbdefb;border-radius:30px;outline:none;font-size:1.05rem;transition:all .3s;background:rgba(255,255,255,.9)
}
#message-input:focus{border-color:#1976d2;box-shadow:0 0 0 4px rgba(25,118,210,.2);background:#fff}
#send-button{
  background:linear-gradient(135deg,#1e88e5,#1976d2);color:#fff;border:none;border-radius:30px;padding:12px 24px;margin-left:12px;cursor:pointer;font-weight:600;font-size:.95rem;
  transition:transform .2s,opacity .2s;box-shadow:0 4px 15px rgba(30,136,229,.4);display:flex;align-items:center;justify-content:center;gap:8px
}
#send-button:hover{opacity:.9;transform:scale(.98)}
#send-button:active{transform:scale(.95)}
@keyframes fadeIn{from{opacity:0;transform:translateY(15px) scale(.9)}to{opacity:1;transform:translateY(0) scale(1)}}
.emoji-popup{
  display:none;position:absolute;bottom:80px;right:20px;background:#fff;border-radius:15px;padding:15px;box-shadow:0 10px 30px rgba(0,0,0,.2);
  z-index:100;width:300px;max-height:250px;overflow-y:auto;border:1px solid #eee
}
.emoji-panel{display:grid;grid-template-columns:repeat(8,1fr);gap:12px}
.emoji-item{font-size:1.8rem;text-align:center;cursor:pointer;padding:8px;border-radius:50%;transition:background .2s,transform .2s}
.emoji-item:hover{background:#e3f2fd;transform:scale(1.2)}
.emoji-toggle{
  position:absolute;right:20px;bottom:85px;background:#fff;border:none;border-radius:50%;width:45px;height:45px;display:flex;align-items:center;justify-content:center;
  cursor:pointer;box-shadow:0 4px 10px rgba(0,0,0,.15);font-size:1.5rem;color:#1e88e5;transition:all .3s
}
.emoji-toggle:hover{background:#e3f2fd;transform:scale(1.1)}
.typing-dots{display:inline-flex;margin-left:10px}
.typing-dots span{display:inline-block;width:8px;height:8px;border-radius:50%;background-color:#1e88e5;margin:0 2px;opacity:.4;animation:dotPulse 1.5s infinite ease-in-out}
.typing-dots span:nth-child(2){animation-delay:.2s}
.typing-dots span:nth-child(3){animation-delay:.4s}
@keyframes dotPulse{0%,100%{transform:scale(.8);opacity:.4}50%{transform:scale(1.2);opacity:.8}}
.quick-emojis-container{padding:10px 0;background:rgba(255,255,255,.8);border-bottom:1px solid #e3f2fd;position:relative;z-index:10}
.quick-emojis-label{display:flex;justify-content:center;margin-bottom:10px;color:#1565c0;font-size:.9rem;font-weight:500}
.emoji-send-desc{font-size:.8rem;color:#757575;text-align:center;margin-top:10px}
.emoji-status{
  position:absolute;top:15px;left:20px;background:rgba(255,255,255,.2);border-radius:50%;width:40px;height:40px;display:flex;align-items:center;justify-content:center;
  box-shadow:0 4px 10px rgba(0,0,0,.1);color:#fff;font-size:1.6rem
}
@media (max-width:600px){
  .chat-container{height:95vh;max-width:100%;border-radius:15px}
  .message{max-width:90%;padding:12px 16px;font-size:1rem}
  #send-button{padding:10px 20px}
  #message-input{padding:12px 18px}
  .emoji-panel{grid-template-columns:repeat(6,1fr)}
  .quick-emoji-btn{font-size:1.1rem;padding:6px}
}
</style>
</head>
<body>
  <div class="chat-container">
    <div class="chat-header">
      <div class="emoji-status">😊</div>
      <i class="fas fa-robot header-icon"></i>
      情感支持机器人（中文 · 表情增强）
    </div>

    <div class="quick-emojis-container">
      <div class="quick-emojis-label">快速发送情绪</div>
      <div class="chat-tools">
        <button class="quick-emoji-btn" data-emoji="😊">😊</button>
        <button class="quick-emoji-btn" data-emoji="😢">😢</button>
        <button class="quick-emoji-btn" data-emoji="😠">😠</button>
        <button class="quick-emoji-btn" data-emoji="❤️">❤️</button>
        <button class="quick-emoji-btn" data-emoji="😨">😨</button>
        <button class="quick-emoji-btn" data-emoji="😲">😲</button>
        <button class="quick-emoji-btn" data-emoji="🤗">🤗</button>
      </div>
    </div>

    <div class="chat-messages" id="chat-messages">
      <div class="welcome-message">
        <div style="margin-bottom:10px;">
          <i class="fas fa-smile-beam" style="font-size:1.6rem;color:#1976d2;margin-bottom:8px;"></i>
          <h3 style="color:#0d47a1;margin-bottom:8px;">你好，我是小Y，你的情感陪伴伙伴</h3>
          <p>用文字或表情告诉我你的心情，我会认真聆听并回应你。</p>
        </div>
        <div style="border-top:1px dashed #90caf9;padding-top:12px;margin-top:10px;font-size:.9rem;">
          <p><strong>小提示：</strong>上方按钮可一键发送常用情绪；点右侧😀可展开更多表情。</p>
        </div>
      </div>
    </div>

    <div class="chat-input-container">
      <div class="chat-input-group">
        <input type="text" id="message-input" placeholder="写下你的感受，或直接输入表情…" autocomplete="off">
        <button class="emoji-toggle" id="emoji-toggle">😀</button>
        <div class="emoji-popup" id="emoji-popup">
          <div class="emoji-panel" id="emoji-panel">
            😠 😡 💢 😤 🤬 😃 😄 😁 🥳 🤩 😂 😅 😇 🤣 🙂 😉 😊
            🥰 😘 😍 ❤️ 💕 💘 💖 💗 💓 💞 🤗 😢 😭 😿 😓 😞 😔
            🥺 😥 😰 😨 😧 😬 😱 👻 😲 😯 🤯 🤔 😐 😑 🙄 🧐
          </div>
          <p class="emoji-send-desc">点击表情即可加入输入框</p>
        </div>
      </div>
      <button id="send-button"><i class="far fa-paper-plane"></i> 发送</button>
    </div>
  </div>

<script>
const messageInput = document.getElementById('message-input');
const sendButton = document.getElementById('send-button');
const chatMessages = document.getElementById('chat-messages');
const emojiToggle = document.getElementById('emoji-toggle');
const emojiPopup = document.getElementById('emoji-popup');
const emojiPanel = document.getElementById('emoji-panel');
const quickEmojiBtns = document.querySelectorAll('.quick-emoji-btn');
const emojiStatus = document.querySelector('.emoji-status');
const chatHeader = document.querySelector('.chat-header');

// 构建表情选择
const emojis = emojiPanel.textContent.split(' ').filter(e => e.trim() !== '');
emojiPanel.innerHTML = '';
emojis.forEach(emoji => {
  const el = document.createElement('div');
  el.className = 'emoji-item';
  el.textContent = emoji;
  el.addEventListener('click', () => {
    messageInput.value += emoji;
    messageInput.focus();
    el.style.backgroundColor = '#e3f2fd';
    setTimeout(()=>{ el.style.backgroundColor=''; }, 300);
    updateEmojiStatus(emoji);
  });
  emojiPanel.appendChild(el);
});

// 快捷发送
quickEmojiBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    const emoji = btn.dataset.emoji;
    sendEmojiDirectly(emoji);
    btn.animate([{transform:'scale(1)'},{transform:'scale(1.5)'},{transform:'scale(1)'}],{duration:400});
    updateEmojiStatus(emoji);
  });
});

function updateEmojiStatus(emoji){
  emojiStatus.textContent = emoji;
  emojiStatus.animate([{transform:'scale(1)'},{transform:'scale(1.4)'},{transform:'scale(1)'}],{duration:500});
  const emotionColors = {'😊':'#4caf50','😢':'#5c6bc0','😠':'#f44336','❤️':'#e91e63','😨':'#795548','😲':'#ff9800','🤗':'#3f51b5'};
  if (emotionColors[emoji]){
    const orig = chatHeader.style.background;
    chatHeader.style.background = emotionColors[emoji];
    setTimeout(()=>{ chatHeader.style.background = orig; }, 2000);
  }
}

const typingIndicator = document.createElement('div');
typingIndicator.className = 'typing-indicator';
typingIndicator.innerHTML = '<i class="fas fa-robot" style="margin-right:10px;color:#1e88e5;"></i> 小Y 正在理解你的情绪… <div class="typing-dots"><span></span><span></span><span></span></div>';

// 安全添加消息（textContent 防 XSS）
function addMessage(text, isUser){
  const el = document.createElement('div');
  el.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
  el.textContent = text;
  chatMessages.appendChild(el);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showTyping(){ typingIndicator.style.display='flex'; chatMessages.appendChild(typingIndicator); chatMessages.scrollTop = chatMessages.scrollHeight; }
function hideTyping(){ typingIndicator.style.display='none'; }

// 情感标签
function showEmotionTag(emotion, emojis){
  const tag = document.createElement('div');
  tag.className = 'emotion-tag';
  const emoIcon = {'愤怒':'😠','快乐':'😄','悲伤':'😢','恐惧':'😨','爱':'❤️','惊讶':'😲','中性':'🤔'}[emotion] || '🤖';
  const text = `识别情绪：${emotion}${emojis ? ' ｜ 表情：' + emojis : ''}`;
  tag.innerHTML = `<span style="font-size:1.1rem;margin-right:10px;">${emoIcon}</span>${text}`;
  chatMessages.appendChild(tag);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  tag.animate([{transform:'translateY(-10px)',opacity:0},{transform:'translateY(0)',opacity:1}],{duration:500,easing:'ease-out'});
}

// 直接发送表情
function sendEmojiDirectly(emoji){
  addMessage(`${emoji}`, true);
  showTyping();
  setTimeout(()=>{
    hideTyping();
    const emotionResponses = {
      '😊': "😄 看到你开心我也很高兴！是什么让你笑起来呢？",
      '😢': "🤗 我感受到你的难过，愿意跟我聊聊发生了什么吗？",
      '😠': "🧘 我能理解你的愤怒，我们一起理一理原因好吗？",
      '❤️': "💖 这份在乎很珍贵，愿意多说一点吗？",
      '😨': "🛡️ 害怕是自然反应，我们一步一步来，好吗？",
      '😲': "🤯 哇，有点意外！发生了什么事？",
      '🤗': "💕 给你一个拥抱！现在的你感觉如何？"
    };
    const resp = emotionResponses[emoji] || "🤔 我看到你在表达情绪，愿意多说一点吗？";
    addMessage(`🤖 ${resp}`, false);

    const map = {'😊':'快乐','😢':'悲伤','😠':'愤怒','❤️':'爱','😨':'恐惧','😲':'惊讶','🤗':'爱'};
    if (map[emoji]) showEmotionTag(map[emoji], emoji);
  }, 800 + Math.random()*800);
}

async function sendMessage(){
  const message = messageInput.value.trim();
  if(!message) return;
  addMessage(`${message}`, true);
  messageInput.value = '';
  messageInput.focus();
  emojiPopup.style.display = 'none';
  showTyping();

  try{
    const r = await fetch('/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({message})});
    if(!r.ok) throw new Error('网络错误：' + r.status);
    const data = await r.json();
    hideTyping();

    if (data.end_conversation){
      addMessage(`🤖 ${data.response}`, false);
      messageInput.disabled = true; sendButton.disabled = true; emojiToggle.style.display = 'none';
    }else{
      addMessage(`🤖 ${data.response}`, false);
      if (data.emotion){ showEmotionTag(data.emotion, data.emojis); }
    }
  }catch(e){
    hideTyping();
    const err = document.createElement('div');
    err.className = 'message bot-message';
    err.textContent = '⚠️ 出了点问题，请稍后再试。';
    chatMessages.appendChild(err);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    console.error(e);
  }
}

sendButton.addEventListener('click', sendMessage);
messageInput.addEventListener('keydown', (e)=>{ if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); sendMessage(); }});

// 切换表情面板与收起逻辑
emojiToggle.addEventListener('click', (e)=>{ e.stopPropagation(); emojiPopup.style.display = (emojiPopup.style.display==='block'?'none':'block'); });
chatMessages.addEventListener('click', ()=>{ emojiPopup.style.display = 'none'; });
document.addEventListener('click', (e)=>{ if(!emojiPopup.contains(e.target) && e.target!==emojiToggle){ emojiPopup.style.display = 'none'; } });

// 机器人图标动画（一次性设置）
const robotIcon = document.querySelector('.fa-robot');
if (robotIcon){
  robotIcon.animate([{transform:'translateY(0)'},{transform:'translateY(-5px)'},{transform:'translateY(0)'}],{duration:2000,iterations:Infinity});
}

// 初始聚焦
window.addEventListener('DOMContentLoaded', ()=>{ messageInput.focus(); });
</script>
</body>
</html>'''
        return html.encode("utf-8"), 200, {"Content-Type": "text/html; charset=utf-8"}
    except Exception as e:
        safe_print(f"❗ 首页加载错误: {e}")
        return ("页面加载出错", 500)


@app.route("/ping")
def ping():
    return jsonify({"status": "alive", "message": "中文情感支持机器人运行中", "version": "2.0-cn"})


# ============================= 启动入口 =============================
if __name__ == "__main__":
    safe_print("🚀 启动 Flask 应用(中文)…")
    port = 8888
    started = False
    max_attempts = 5

    for attempt in range(max_attempts):
        try:
            safe_print(f"🔌 尝试在端口 {port} 启动服务...")
            app.run(debug=False, host="0.0.0.0", port=port, use_reloader=False)
            started = True
            break
        except OSError as e:
            safe_print(f"❌ 端口 {port} 启动失败: {str(e)}")
            if getattr(e, "errno", None) in {errno.EADDRINUSE, 98, 48, 10048}:
                safe_print(f"🔄 端口 {port} 已被占用，尝试新端口")
                port += 1
            else:
                safe_print(f"⚠️ 启动错误: {str(e)}")
                break

    if not started:
        safe_print(f"\n⛔ 无法启动服务，尝试端口范围 ({port - max_attempts} 到 {port}) 不可用")
        safe_print("🛠️ 请关闭占用端口的程序或手动指定端口：python app_zh.py --port=YOUR_PORT_NUMBER")
