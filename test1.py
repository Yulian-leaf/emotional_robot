# app.py
# -*- coding: utf-8 -*-
"""
Emotional Support Bot (Emoji Enhanced) - Fixed & Hardened
---------------------------------------------------------
依赖(建议版本):
    pip install "flask>=2.2" "transformers>=4.40" "torch>=2.2" emoji

启动:
    python app.py
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
# 使用国内镜像（如不需要可注释）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

app = Flask(__name__)
# 保证 JSON 返回不转义中文和表情
app.config["JSON_AS_ASCII"] = False


# ----------------------------- 安全响应头(轻量) -----------------------------
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


class EmotionalSupportBot:
    def __init__(self):
        safe_print("😊 初始化情感分析模型...")

        device = 0 if torch.cuda.is_available() else -1

        # 情感分析模型
        self.emotion_classifier = pipeline(
            "text-classification",
            model="bhadresh-savani/distilbert-base-uncased-emotion",
            return_all_scores=True,
            device=device,
        )

        safe_print("💬 初始化响应生成模型...")
        # 文本生成模型
        self.response_generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            device=device,
        )

        # 预设共情回应
        self.empathy_responses = {
            "sadness": [
                "😢 I hear that you're feeling down. I understand that feeling.",
                "🤗 Would you like to share more? I care about how you're feeling.",
                "💔 Feeling sad can be really tough, but remember these emotions are temporary.",
                "🤝 I know you're in pain right now, but please remember you're not alone.",
                "🌧️ Sometimes it's necessary to allow yourself to feel sad. I'm here with you.",
            ],
            "joy": [
                "🎉 I'm so happy for you! These beautiful moments are worth cherishing.",
                "😄 That's wonderful! Could you tell me what made you so happy?",
                "🥰 Hearing this makes me happy too!",
                "🌈 This joyful feeling is so nice! Would you like to share more?",
                "☀️ It's heartwarming to see you happy.",
            ],
            "anger": [
                "😠 I understand you must be feeling angry right now, and that's valid. This is a strong emotion, but what matters most is how we handle it.",
                "💢 Anger can be really troubling. Would you like to talk about what caused it?",
                "🧘 Take a deep breath and try to relax a little, okay?",
                "⚡ When we're angry, it's hard to think clearly. Can I help you sort through your thoughts?",
                "💥 Anger can be really troubling. Would you like to talk about what caused it?",
            ],
            "fear": [
                "😨 I sense you might be feeling uneasy. Fear is a natural human emotion.",
                "😰 When we're afraid, we often feel most alone, but remember you're not alone.",
                "👣 Sometimes the best way to face fear is to take small steps forward.",
                "🛡️ Fear is our brain's way of protecting us, though sometimes it overprotects.",
                "🤝 I'm here, and we can face what scares you together.",
            ],
            "love": [
                "💖 It's beautiful to feel love, whether for others or for yourself.",
                "❤️ To love and be loved are among life's most precious experiences.",
                "💓 When we feel love, the whole world seems different.",
                "💕 Love truly gives life special meaning.",
                "💞 It's so heartwarming to hear you share about love in your life.",
            ],
            "surprise": [
                "😲 How unexpected! Could you tell me what happened?",
                "🎊 Life is full of surprises. What surprised you this time?",
                "🌀 Unexpected events can sometimes give us new perspectives.",
                "🎯 Sometimes surprises can become turning points. What do you think?",
                "✨ Wow! Could you tell me what surprised you so much?",
            ],
            "neutral": [
                "😌 I'm here to listen. Could you tell me more about how you're feeling?",
                "💬 What else would you like to share?",
                "🤔 That's interesting. Could you elaborate?",
                "🌱 Every experience helps us grow. Would you like to talk more about this?",
                "📝 I'm taking notes. Feel free to share anything on your mind.",
            ],
        }

        # 表情与情感映射（修正了错误条目，并兼容 ❤）
        self.emoji_emotion_map = {
            "😠": "anger",
            "😡": "anger",
            "💢": "anger",
            "😤": "anger",
            "🤬": "anger",
            "😃": "joy",
            "😄": "joy",
            "😁": "joy",
            "🥳": "joy",
            "🤩": "joy",
            "😂": "joy",
            "😅": "joy",
            "😇": "joy",
            "🤣": "joy",       # 修正
            "🙂": "joy",
            "😉": "joy",
            "😊": "joy",
            "🥰": "love",
            "😘": "love",
            "😍": "love",
            "❤️": "love",
            "❤": "love",       # 兼容无 VS-16 的心形
            "💕": "love",
            "💘": "love",
            "💖": "love",
            "💗": "love",
            "💓": "love",
            "💞": "love",
            "🤗": "love",
            "😢": "sadness",
            "😭": "sadness",
            "😿": "sadness",
            "😓": "sadness",    # 修正
            "😞": "sadness",
            "😔": "sadness",
            "🥺": "sadness",
            "😥": "sadness",
            "😰": "fear",
            "😨": "fear",
            "😧": "fear",
            "😬": "fear",
            "😱": "fear",
            "👻": "fear",
            "😲": "surprise",
            "😯": "surprise",
            "🤯": "surprise",
            "🤔": "neutral",
            "😐": "neutral",
            "😑": "neutral",
            "🙄": "neutral",
            "🧐": "neutral",
        }

        # 表情描述
        self.emoji_description = {
            "😠": "angry face",
            "😡": "pouting face",
            "😃": "smiling face",
            "😄": "smiling face with open mouth",
            "😁": "grinning face",
            "🥳": "partying face",
            "🤩": "star-struck face",
            "😂": "laughing with tears",
            "😅": "sweating smile",
            "😇": "smiling face with halo",
            "🤣": "rolling on the floor laughing",
            "🙂": "slight smile",
            "😉": "winking face",
            "😊": "smiling face with smiling eyes",
            "🥰": "smiling face with hearts",
            "😘": "face blowing kiss",
            "😍": "heart eyes",
            "❤️": "red heart",
            "❤": "red heart",
            "💕": "two hearts",
            "💘": "heart with arrow",
            "💖": "sparkling heart",
            "💗": "growing heart",
            "💓": "beating heart",
            "💞": "revolving hearts",
            "🤗": "hugging face",
            "😢": "crying face",
            "😭": "loudly crying face",
            "😿": "crying cat",
            "😓": "downcast face with sweat",
            "😞": "disappointed face",
            "😔": "pensive face",
            "🥺": "pleading face",
            "😥": "sad but relieved face",
            "😰": "anxious face with sweat",
            "😨": "fearful face",
            "😧": "anguished face",
            "😬": "grimacing face",
            "😱": "face screaming in fear",
            "👻": "ghost",
            "😲": "astonished face",
            "😯": "hushed face",
            "🤯": "exploding head",
            "🤔": "thinking face",
            "😐": "neutral face",
            "😑": "expressionless face",
            "🙄": "face with rolling eyes",
            "🧐": "face with monocle",
        }

        safe_print("🤖 情感支持机器人初始化完成")

    def detect_emotion(self, user_input: str):
        if not user_input.strip():
            return None, ""

        # 兼容多码点表情的提取（如 ❤️）
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

        emoji_descriptions = [self.emoji_description.get(c, "emoji") for c in emojis]
        detected_emojis = ""

        try:
            emotions = self.emotion_classifier(user_input)[0]
            primary = max(emotions, key=lambda x: x["score"])
            base_emotion, base_score = primary["label"], primary["score"]

            if emojis:
                from collections import Counter

                cnt = Counter(self.emoji_emotion_map.get(e) for e in emojis if e in self.emoji_emotion_map)
                if cnt:
                    emoji_emotion, freq = cnt.most_common(1)[0]
                    emoji_confidence = freq / max(1, len(emojis))
                    final_emotion = emoji_emotion if (emoji_confidence > 0.5 and base_score < 0.7) else base_emotion
                else:
                    final_emotion = base_emotion
            else:
                final_emotion = base_emotion

            detected_emojis = " ".join(emoji_descriptions) if emoji_descriptions else "None"
            safe_print(f"🎭 检测到的情感: {final_emotion} (文本: {base_emotion}, 表情: {detected_emojis})")
            return final_emotion, detected_emojis
        except Exception as e:
            safe_print(f"❗ 情感分析出错: {e}")
            detected_emojis = " ".join(emoji_descriptions) if emoji_descriptions else "None"
            return "neutral", detected_emojis

    def generate_response(self, user_input: str, detected_emotion: str, detected_emojis: str):
        # 优先使用预设共情回复
        if detected_emotion in self.empathy_responses:
            response = random.choice(self.empathy_responses[detected_emotion])
            safe_print(f"💡 使用预设回应: {response}")
            return response

        # 构建提示
        emoji_context = f"(noting that you used {detected_emojis})" if detected_emojis != "None" else ""
        prompt = (
            f"Respond to this statement in a warm and friendly tone, showing care and understanding "
            f"{emoji_context}. User says: {user_input}"
        )
        safe_print(f"💭 生成回应提示: {prompt}")

        try:
            generated = self.response_generator(
                prompt,
                max_new_tokens=80,
                do_sample=True,       # 启用采样，使 temperature 生效
                temperature=0.7,
                top_p=0.95,
                num_return_sequences=1,
            )
            response = generated[0]["generated_text"]
            safe_print(f"🤖 生成的回应: {response}")

            # 确保回应中含有表情符号(若无且有情感类型)
            if not any(ch in response for ch in "😀😁😂😃😄😅😆😉😊😋😎😍😘😗😙😚😇😐😑😶😏😣😥😮"):
                emotion_emojis = {
                    "anger": "😠",
                    "joy": "😄",
                    "sadness": "😢",
                    "fear": "😨",
                    "love": "❤️",
                    "surprise": "😲",
                    "neutral": "🤔",
                }
                emoji = emotion_emojis.get(detected_emotion, "🤖")
                response = f"{emoji} {response}"

            return response
        except Exception as e:
            safe_print(f"❗ 生成回应出错: {e}")
            return "😕 I sense your emotions but I'm not sure how to respond. Could you tell me more?"

    def validate_response(self, response: str, user_input: str):
        low = response.lower()
        if any(w in low for w in ["inappropriate", "negative", "harmful"]):
            return "🙏 I'm sorry, I'm not sure how to respond appropriately. Could we talk about something else?"
        if "don't understand" in low or "don't know" in low:
            return f"📚 I'm still learning to better understand human emotions. You said '{user_input}', could you explain more?"
        return response


safe_print("🤖 启动情感支持机器人...")
bot = EmotionalSupportBot()


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
                    "response": "😶 I didn't quite catch that. Could you repeat?",
                    "emotion": "neutral",
                    "emojis": "",
                    "end_conversation": False,
                }
            )

        if re.search(r"\b(bye|goodbye|exit|quit)\b", user_input, re.I):
            safe_print("👋 结束对话请求")
            return jsonify(
                {
                    "response": "💖 Thank you for sharing! Remember, you're not alone. I'm here whenever you need me.",
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
        safe_print(f"❗ 聊天请求处理错误: {e}")
        return jsonify(
            {
                "response": "😓 I'm having some trouble processing your request. Could you try again?",
                "emotion": "neutral",
                "emojis": "",
                "end_conversation": False,
            }
        )


@app.route("/")
def index():
    """主页面服务路由 - 内嵌 HTML（已修复 CSS/JS 与 XSS 问题）"""
    try:
        html_content = r'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Emotional Support Bot - Xiao Y</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        body {
            background: linear-gradient(135deg, #1e88e5, #1976d2);
            display: flex; justify-content: center; align-items: center;
            min-height: 100vh; padding: 20px; color: #333;
        }
        .chat-container {
            width: 100%; max-width: 500px; height: 90vh;
            background-color: rgba(255, 255, 255, 0.95);
            border-radius: 20px; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            display: flex; flex-direction: column; overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .chat-header {
            background: linear-gradient(135deg, #1e88e5, #1976d2); color: white;
            padding: 20px; text-align: center; font-size: 1.4rem; font-weight: 600; position: relative;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2); border-bottom: 2px solid rgba(255, 255, 255, 0.2);
        }
        .header-icon { font-size: 2rem; margin-right: 10px; vertical-align: middle; }
        .chat-messages {
            flex: 1; padding: 20px; overflow-y: auto; background: rgba(235, 245, 255, 0.7);
            display: flex; flex-direction: column;
        }
        .welcome-message {
            text-align: center; margin-bottom: 20px; color: #0d47a1; font-size: 1rem; line-height: 1.6;
            background: rgba(255, 255, 255, 0.9); padding: 15px; border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1); border: 1px solid rgba(25, 118, 210, 0.2);
        }
        .message {
            max-width: 85%; padding: 15px 20px; margin-bottom: 15px; border-radius: 20px; line-height: 1.5; position: relative;
            animation: fadeIn 0.4s ease; font-size: 1.1rem;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* 修复 */
            transition: transform 0.3s, box-shadow 0.3s; overflow-wrap: break-word; border: 1px solid rgba(0, 0, 0, 0.05);
        }
        .user-message {
            background: linear-gradient(135deg, #1e88e5, #1976d2); color: white; margin-left: auto; border-bottom-right-radius: 5px;
            box-shadow: 0 4px 10px rgba(30, 136, 229, 0.3); transform-origin: right; text-align: right;
        }
        .bot-message {
            background: linear-gradient(135deg, #ffffff, #f8f9fa); color: #333; margin-right: auto; border-bottom-left-radius: 5px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08); transform-origin: left; text-align: left;
        }
        .message:hover { transform: translateY(-3px); box-shadow: 0 6px 15px rgba(0, 0, 0, 0.15); }
        .typing-indicator {
            display: none; padding: 15px 20px; background: #f8f9fa; color: #333; border-radius: 20px; margin-bottom: 15px; width: fit-content;
            border-bottom-left-radius: 5px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08); font-size: 1.1rem;
        }
        .emotion-tag {
            font-size: 0.85rem; color: #1976d2; margin-top: 8px; font-weight: 500; text-align: left; display: flex; align-items: center;
            padding: 8px 15px; background: rgba(30, 136, 229, 0.1); border-radius: 15px; margin-left: auto; margin-right: auto; width: fit-content;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        }
        .chat-input-container { display: flex; padding: 15px 20px; background: white; border-top: 1px solid rgba(0, 0, 0, 0.1); }
        .chat-input-group { flex: 1; display: flex; position: relative; }
        .chat-tools {
            display: flex; align-items: center; margin-bottom: 10px; margin-left: 10px; background: rgba(30, 136, 229, 0.05);
            padding: 8px 15px; border-radius: 25px;
        }
        .quick-emoji-btn {
            font-size: 1.4rem; background: none; border: none; cursor: pointer; margin: 0 5px; padding: 8px; border-radius: 50%;
            transition: all 0.3s;
        }
        .quick-emoji-btn:hover { background: rgba(30, 136, 229, 0.15); transform: scale(1.15); }
        #message-input {
            flex: 1; padding: 15px 20px; border: 2px solid #bbdefb; border-radius: 30px; outline: none; font-size: 1.1rem; transition: all 0.3s;
            background: rgba(255, 255, 255, 0.9);
        }
        #message-input:focus { border-color: #1976d2; box-shadow: 0 0 0 4px rgba(25, 118, 210, 0.2); background: white; }
        #send-button {
            background: linear-gradient(135deg, #1e88e5, #1976d2); color: white; border: none; border-radius: 30px; padding: 15px 30px; margin-left: 15px;
            cursor: pointer; font-weight: 600; font-size: 1rem; transition: transform 0.2s, opacity 0.2s;
            box-shadow: 0 4px 15px rgba(30, 136, 229, 0.4); display: flex; align-items: center; justify-content: center; gap: 10px;
        }
        #send-button:hover { opacity: 0.9; transform: scale(0.98); }
        #send-button:active { transform: scale(0.95); }

        @keyframes fadeIn { from { opacity: 0; transform: translateY(15px) scale(0.9); } to { opacity: 1; transform: translateY(0) scale(1); } }
        @keyframes floatIcon { 0% { transform: translateY(0); } 50% { transform: translateY(-5px); } 100% { transform: translateY(0); } }

        .emoji-popup {
            display: none; position: absolute; bottom: 80px; right: 20px; background: white; border-radius: 15px; padding: 15px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2); z-index: 100; width: 300px; max-height: 250px; overflow-y: auto; border: 1px solid #eee;
        }
        .emoji-panel { display: grid; grid-template-columns: repeat(8, 1fr); gap: 12px; }
        .emoji-item {
            font-size: 1.8rem; text-align: center; cursor: pointer; padding: 8px; border-radius: 50%;
            transition: background 0.2s, transform 0.2s; /* 修复 */
        }
        .emoji-item:hover { background: #e3f2fd; transform: scale(1.2); }

        .emoji-toggle {
            position: absolute; right: 20px; bottom: 85px; background: white; border: none; border-radius: 50%; width: 45px; height: 45px;
            display: flex; align-items: center; justify-content: center; cursor: pointer; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
            font-size: 1.5rem; color: #1e88e5; transition: all 0.3s;
        }
        .emoji-toggle:hover { background: #e3f2fd; transform: scale(1.1); }

        .typing-dots { display: inline-flex; margin-left: 10px; }
        .typing-dots span { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background-color: #1e88e5; margin: 0 2px; opacity: 0.4;
            animation: dotPulse 1.5s infinite ease-in-out; }
        .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
        .typing-dots span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes dotPulse { 0%, 100% { transform: scale(0.8); opacity: 0.4; } 50% { transform: scale(1.2); opacity: 0.8; } }

        .quick-emojis-container { padding: 10px 0; background: rgba(255, 255, 255, 0.8); border-bottom: 1px solid #e3f2fd; position: relative; z-index: 10; }
        .quick-emojis-label { display: flex; justify-content: center; margin-bottom: 10px; color: #1565c0; font-size: 0.9rem; font-weight: 500; }
        .emoji-send-desc { font-size: 0.8rem; color: #757575; text-align: center; margin-top: 10px; }
        .emoji-status {
            position: absolute; top: 15px; left: 20px; background: rgba(255, 255, 255, 0.2); border-radius: 50%; width: 40px; height: 40px;
            display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1); color: white; font-size: 1.8rem;
        }

        @media (max-width: 600px) {
            .chat-container { height: 95vh; max-width: 100%; border-radius: 15px; }
            .message { max-width: 90%; padding: 12px 16px; font-size: 1rem; }
            #send-button { padding: 12px 25px; }
            #message-input { padding: 12px 18px; }
            .emoji-panel { grid-template-columns: repeat(6, 1fr); }
            .quick-emoji-btn { font-size: 1.1rem; padding: 6px; }
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <div class="emoji-status">😊</div>
            <i class="fas fa-robot header-icon"></i>
            Emotional Support Bot (Emoji Enhanced)
        </div>

        <div class="quick-emojis-container">
            <div class="quick-emojis-label">Quick Send Emotions</div>
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
                <div style="margin-bottom: 15px;">
                    <i class="fas fa-smile-beam" style="font-size: 2rem; color: #1976d2; margin-bottom: 10px;"></i>
                    <h3 style="color: #0d47a1; margin-bottom: 10px;">Hello! I'm Xiao Y, your emotional support companion</h3>
                    <p>I'm here to listen and support you with advanced emoji recognition.<br>Express your feelings with text or emojis - I'll understand!</p>
                </div>
                <div style="border-top: 1px dashed #90caf9; padding-top: 15px; margin-top: 10px; font-size: 0.9rem;">
                    <p><strong>Tip:</strong> Click on the quick emoji buttons above to send emotions instantly, or use the 😀 button for more emojis!</p>
                </div>
            </div>
        </div>

        <div class="chat-input-container">
            <div class="chat-input-group">
                <input type="text" id="message-input" placeholder="Express yourself with words or emojis..." autocomplete="off">
                <button class="emoji-toggle" id="emoji-toggle">😀</button>
                <div class="emoji-popup" id="emoji-popup">
                    <div class="emoji-panel" id="emoji-panel">
                        😠 😡 💢 😤 🤬 😃 😄 😁 🥳 🤩 😂 😅 😇 🤣 🙂 😉 😊
                        🥰 😘 😍 ❤️ 💕 💘 💖 💗 💓 💞 🤗 😢 😭 😿 😓 😞 😔
                        🥺 😥 😰 😨 😧 😬 😱 👻 😲 😯 🤯 🤔 😐 😑 🙄 🧐
                    </div>
                    <p class="emoji-send-desc">Click any emoji to add to your message</p>
                </div>
            </div>
            <button id="send-button">
                <i class="far fa-paper-plane"></i>
                Send
            </button>
        </div>
    </div>

    <script>
        // DOM
        const messageInput = document.getElementById('message-input');
        const sendButton = document.getElementById('send-button');
        const chatMessages = document.getElementById('chat-messages');
        const emojiToggle = document.getElementById('emoji-toggle');
        const emojiPopup = document.getElementById('emoji-popup');
        const emojiPanel = document.getElementById('emoji-panel');
        const quickEmojiBtns = document.querySelectorAll('.quick-emoji-btn');
        const emojiStatus = document.querySelector('.emoji-status');
        const chatHeader = document.querySelector('.chat-header');

        // 构建表情选择项
        const emojis = emojiPanel.textContent.split(' ').filter(e => e.trim() !== '');
        emojiPanel.innerHTML = '';
        emojis.forEach(emoji => {
            const emojiElement = document.createElement('div');
            emojiElement.className = 'emoji-item';
            emojiElement.textContent = emoji;
            emojiElement.addEventListener('click', () => {
                messageInput.value += emoji;
                messageInput.focus();
                emojiElement.style.backgroundColor = '#e3f2fd';
                setTimeout(() => { emojiElement.style.backgroundColor = ''; }, 300);
                updateEmojiStatus(emoji);
            });
            emojiPanel.appendChild(emojiElement);
        });

        // 快捷发送
        quickEmojiBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const emoji = btn.dataset.emoji;
                sendEmojiDirectly(emoji);
                btn.animate([{ transform: 'scale(1)' }, { transform: 'scale(1.5)' }, { transform: 'scale(1)' }], { duration: 400 });
                updateEmojiStatus(emoji);
            });
        });

        function updateEmojiStatus(emoji) {
            emojiStatus.textContent = emoji;
            emojiStatus.animate([{ transform: 'scale(1)' }, { transform: 'scale(1.4)' }, { transform: 'scale(1)' }], { duration: 500 });

            const emotionColors = {
                '😊': '#4caf50',
                '😢': '#5c6bc0',
                '😠': '#f44336',
                '❤️': '#e91e63',
                '😨': '#795548',
                '😲': '#ff9800',
                '🤗': '#3f51b5'
            };
            if (emotionColors[emoji]) {
                const origBg = chatHeader.style.background;
                chatHeader.style.background = emotionColors[emoji];
                setTimeout(() => { chatHeader.style.background = origBg; }, 2000);
            }
        }

        // 直接发送表情
        function sendEmojiDirectly(emoji) {
            addMessage(`${emoji}`, true);
            showTypingIndicator();
            setTimeout(() => {
                hideTypingIndicator();
                const emotionResponses = {
                    '😊': "😄 It's wonderful to see you happy! What's making you smile today?",
                    '😢': "🤗 I sense you're feeling down. Would you like to share what's troubling you?",
                    '😠': "🧘‍♀️ I feel your anger. What's causing this frustration? I'm here to listen.",
                    '❤️': "💖 Love is a beautiful emotion. Would you like to share more about this feeling?",
                    '😨': "🛡️ Fear can be overwhelming. What concerns you right now?",
                    '😲': "🤯 Wow, what a surprise! What happened?",
                    '🤗': "💕 Hugs sent your way! How are you feeling today?"
                };
                const response = emotionResponses[emoji] || "🤔 I see you're feeling something. Would you like to share more?";
                addMessage(`🤖 ${response}`, false);

                const emotionMap = { '😊': 'joy', '😢': 'sadness', '😠': 'anger', '❤️': 'love', '😨': 'fear', '😲': 'surprise', '🤗': 'support' };
                if (emotionMap[emoji]) showEmotionTag(emotionMap[emoji], emoji);
            }, 1000 + Math.random() * 1000);
        }

        // 切换表情面板
        emojiToggle.addEventListener('click', (e) => {
            e.stopPropagation();
            emojiPopup.style.display = emojiPopup.style.display === 'block' ? 'none' : 'block';
        });

        // 打字指示器
        const typingIndicator = document.createElement('div');
        typingIndicator.className = 'typing-indicator';
        typingIndicator.innerHTML = '<i class="fas fa-robot" style="margin-right: 10px; color: #1e88e5;"></i> Xiao Y is analyzing your emotions... <div class="typing-dots"><span></span><span></span><span></span></div>';

        // **安全**添加消息（防 XSS：使用 textContent 而非 innerHTML）
        function addMessage(text, isUser) {
            const messageElement = document.createElement('div');
            messageElement.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
            messageElement.textContent = text; // 安全
            chatMessages.appendChild(messageElement);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        function showTypingIndicator() {
            typingIndicator.style.display = 'flex';
            chatMessages.appendChild(typingIndicator);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        function hideTypingIndicator() {
            typingIndicator.style.display = 'none';
        }

        function showEmotionTag(emotion, emojis) {
            const emotionTag = document.createElement('div');
            emotionTag.className = 'emotion-tag';
            const emotionEmojis = { 'anger': '😠', 'joy': '😄', 'sadness': '😢', 'fear': '😨', 'love': '❤️', 'surprise': '😲', 'support': '🤗', 'neutral': '🤔' };
            const emoji = emotionEmojis[emotion] || '🤖';
            const text = `Detected emotion: ${emotion}${emojis ? ' | Sent emoji: ' + emojis : ''}`;
            emotionTag.innerHTML = `<span style="font-size: 1.2rem; margin-right: 10px;">${emoji}</span>${text}`;
            chatMessages.appendChild(emotionTag);
            chatMessages.scrollTop = chatMessages.scrollHeight;
            emotionTag.animate([{ transform: 'translateY(-10px)', opacity: 0 }, { transform: 'translateY(0)', opacity: 1 }], { duration: 500, easing: 'ease-out' });
        }

        async function sendMessage() {
            const message = messageInput.value.trim();
            if (!message) return;

            addMessage(`${message}`, true);
            messageInput.value = '';
            messageInput.focus();
            emojiPopup.style.display = 'none';
            showTypingIndicator();

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message })
                });
                if (!response.ok) throw new Error(`Server responded with status ${response.status}`);
                const data = await response.json();
                hideTypingIndicator();

                if (data.end_conversation) {
                    addMessage(`🤖 ${data.response}`, false);
                    messageInput.disabled = true;
                    sendButton.disabled = true;
                    emojiToggle.style.display = 'none';
                } else {
                    addMessage(`🤖 ${data.response}`, false);
                    if (data.emotion) showEmotionTag(data.emotion, data.emojis);
                }
            } catch (error) {
                hideTypingIndicator();
                const err = document.createElement('div');
                err.className = 'message bot-message';
                err.textContent = '⚠️ Sorry, I encountered a problem. Could you try again?';
                chatMessages.appendChild(err);
                chatMessages.scrollTop = chatMessages.scrollHeight;
                console.error('Error:', error);
            }
        }

        sendButton.addEventListener('click', sendMessage);
        messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
        });

        chatMessages.addEventListener('click', () => { emojiPopup.style.display = 'none'; });
        document.addEventListener('click', (e) => {
            if (!emojiPopup.contains(e.target) && e.target !== emojiToggle) emojiPopup.style.display = 'none';
        });
        window.addEventListener('DOMContentLoaded', () => { messageInput.focus(); });

        // 机器人图标动画：启动一次，无内存泄露
        const robotIcon = document.querySelector('.fa-robot');
        if (robotIcon) {
            robotIcon.animate(
                [{ transform: 'translateY(0)' }, { transform: 'translateY(-5px)' }, { transform: 'translateY(0)' }],
                { duration: 2000, iterations: Infinity }
            );
        }
    </script>
</body>
</html>'''
        return html_content.encode("utf-8"), 200, {"Content-Type": "text/html; charset=utf-8"}
    except Exception as e:
        safe_print(f"❗ 首页加载错误: {e}")
        error_html = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Error Page</title>
            <style>
                body {{ font-family: Arial, sans-serif; background-color: #f0f0f0; padding: 20px; text-align: center; }}
                .error-container {{ max-width: 600px; margin: 50px auto; padding: 20px; background: white; border-radius: 10px;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1); }}
                h1 {{ color: #d32f2f; }}
                .debug-info {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; text-align: left; margin-top: 20px;
                    font-family: monospace; font-size: 14px; overflow-x: auto; }}
            </style>
        </head>
        <body>
            <div class="error-container">
                <h1>Emotional Support Bot</h1>
                <p>The bot is running properly but experienced a display issue.</p>
                <p>You can interact with the bot by sending POST requests to /chat endpoint.</p>
                <div class="debug-info"><strong>Debug information:</strong><p>{str(e)}</p></div>
                <div style="margin-top: 30px;">
                    <p><strong>To troubleshoot:</strong></p>
                    <ul style="text-align: left;">
                        <li>Ensure the application has permission to access resources</li>
                        <li>Check your network connection if models need to be downloaded</li>
                        <li>Restart the application</li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        '''
        return error_html.encode("utf-8"), 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/ping")
def ping():
    return jsonify(
        {
            "status": "alive",
            "message": "Enhanced Emotional Support Bot is running",
            "version": "2.0",
            "features": ["emoji_detection", "anger_support"],
        }
    )


# ============================= 启动入口 =============================
if __name__ == "__main__":
    safe_print("🚀 启动 Flask 应用...")
    port = 8888
    started = False
    max_attempts = 5

    for attempt in range(max_attempts):
        try:
            safe_print(f"🔌 尝试在端口 {port} 启动服务...")
            # 生产建议将 debug=False；use_reloader=False 避免多进程重复加载
            app.run(debug=False, host="0.0.0.0", port=port, use_reloader=False)
            started = True
            break
        except OSError as e:
            safe_print(f"❌ 端口 {port} 启动失败: {str(e)}")
            # 使用 errno 判断端口占用，兼容多平台
            if getattr(e, "errno", None) in {errno.EADDRINUSE, 98, 48, 10048}:
                safe_print(f"🔄 端口 {port} 已被占用，尝试新端口")
                port += 1
            else:
                safe_print(f"⚠️ 启动错误: {str(e)}")
                break

    if not started:
        safe_print(f"\n⛔ 无法启动服务，尝试端口范围 ({port - max_attempts} 到 {port}) 均不可用")
        safe_print("🛠️ 请关闭其他程序或指定端口: python app.py --port=YOUR_PORT_NUMBER")
