from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Initialize the sentiment pipeline using a multilingual sentiment model.
# The model: cardiffnlp/twitter-xlm-roberta-base-sentiment
# This model outputs labels: "negative", "neutral", "positive" with scores.
model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
sentiment_pipeline = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)

# Store user sessions
# { user_id: { "state": str, "relationship": str, "method": str, "history": [(user_msg, response)], "emotion_history": [(msg, score)], "feedback": [] } }
session_data = {}

# Define sentiment range to category mapping
def sentiment_category(score):
    if score <= -0.5:
        return "very_negative"
    elif -0.5 < score < 0.0:
        return "negative"
    elif 0.0 <= score <= 0.5:
        return "neutral"
    else:
        return "positive"

def analyze_sentiment(text):
    # Use the hugging face pipeline
    results = sentiment_pipeline(text)
    # results is a list, e.g. [{'label': 'positive', 'score': 0.99}]
    result = results[0]
    label = result['label']
    score = result['score']
    # Map label to a continuous scale:
    if label == "negative":
        sentiment_score = -1.0 * score
    elif label == "neutral":
        sentiment_score = 0.0
    else:  # positive
        sentiment_score = 1.0 * score
    return sentiment_score

# Responses now vary by four categories: very_negative, negative, neutral, positive
# We show only a subset of states for brevity; expand similarly for all states.
RESPONSES = {
    "INIT": {
        "en": {
            "very_negative": (
                "I sense this might be really troubling you, and it's okay to feel that way. "
                "Let's go step by step. Could you tell me about your relationship with this person?"
            ),
            "negative": (
                "It seems you might be feeling a bit down, I understand. Could you share your relationship with them, like are they a friend, partner, or colleague?"
            ),
            "neutral": (
                "I see you're curious about what's going on. To help me understand better, can you tell me what your relationship is with this person?"
            ),
            "positive": (
                "I'm glad you're open to discussing this. Could you tell me what kind of relationship you share with them?"
            )
        },
        "zh": {
            "very_negative": (
                "我感觉到这件事让你很难过，这很正常。"
                "我们可以慢慢来。能告诉我你们的关系吗？例如：朋友、恋人或同事？"
            ),
            "negative": (
                "我能看出你可能有点低落。能告诉我你们是什么关系吗？例如朋友、伴侣或同事？"
            ),
            "neutral": (
                "我明白你的疑惑。为了更好帮助你，能先告诉我你们之间的关系吗？"
            ),
            "positive": (
                "很高兴你愿意讨论。能告诉我你们之间是什么关系吗？例如朋友、伴侣或同事？"
            )
        },
        "ja": {
            "very_negative": (
                "とても辛い状況のようですね、それは自然なことです。"
                "ゆっくり進めましょう。相手との関係を教えていただけますか？"
            ),
            "negative": (
                "少し落ち込んでいるように感じます。相手との関係は、友人、恋人、同僚など、どのようなものですか？"
            ),
            "neutral": (
                "気になっているのですね。状況を理解するため、相手との関係を教えていただけますか？"
            ),
            "positive": (
                "この件について話してくれてうれしいです。相手との関係はどのようなものですか？"
            )
        }
    },
    # ... Similarly define for other states (ASK_DETAILS, NO_INFO, HAS_INFO, SUGGEST_STRATEGY, END)
    # For brevity, let's reuse simpler logic: in a real scenario, you'd define all states similarly.
    "END": {
        "en": {
            "very_negative": (
                "I know this might still feel heavy. If you want to talk more or explore other angles, I’ll be here."
            ),
            "negative": (
                "I hope these suggestions help even a little. You can come back anytime if you'd like to chat more."
            ),
            "neutral": (
                "I hope these ideas help. Feel free to return if you want to discuss more."
            ),
            "positive": (
                "I hope these suggestions were helpful. If you need more guidance, I'll be here."
            )
        },
        "zh": {
            "very_negative": (
                "我知道这可能仍让你感到沉重。如果你想再谈或从不同角度看待，我都在这儿。"
            ),
            "negative": (
                "希望这些建议能稍微帮助到你。如果你想再聊，随时回来。"
            ),
            "neutral": (
                "希望这些想法对你有帮助。如果你想继续讨论，欢迎随时回来。"
            ),
            "positive": (
                "希望这些建议对你有所帮助。如果你需要更多指导，我随时在这里。"
            )
        },
        "ja": {
            "very_negative": (
                "まだ重い気持ちがあるかもしれません。別の視点から話してみたくなったら、いつでもお越しください。"
            ),
            "negative": (
                "これらの提案が少しでも役立つことを願っています。もっと話したくなったらいつでもどうぞ。"
            ),
            "neutral": (
                "このアイデアがお役に立てば幸いです。再度話したくなったら、またお待ちしています。"
            ),
            "positive": (
                "これらの提案が参考になればうれしいです。さらにアドバイスが必要なときはいつでも来てください。"
            )
        }
    }
}

def get_response(state, lang, sentiment_score):
    # Determine category
    cat = sentiment_category(sentiment_score)
    state_responses = RESPONSES.get(state, {})
    lang_responses = state_responses.get(lang, {})
    # If this state or language not defined, fallback to English neutral
    if not lang_responses:
        lang_responses = RESPONSES.get(state, {}).get("en", {})
    response = lang_responses.get(cat, "")
    if not response:
        # fallback to neutral if nothing found
        response = lang_responses.get("neutral", "I’m here to listen.")
    return response

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_id = data.get("user_id", "default_user")
    user_message = data.get("user_message", "").strip()
    lang = data.get("lang", "en")  # Default language

    if user_id not in session_data:
        session_data[user_id] = {
            "state": "INIT",
            "relationship": None,
            "method": None,
            "history": [],
            "emotion_history": [],
            "feedback": []
        }

    user_state = session_data[user_id]["state"]

    # Special case: If we are asking for feedback
    if user_state == "FEEDBACK":
        # Check if user gave feedback
        if user_message in ["👍", "👎"]:
            session_data[user_id]["feedback"].append(user_message)
            # Provide a 'thank you' response
            if lang == "en":
                response_text = "Thanks for your feedback! If you need more help, just ask."
            elif lang == "zh":
                response_text = "谢谢你的反馈！如果你需要更多帮助，请随时告诉我。"
            else:  # ja
                response_text = "フィードバックありがとうございます！また何かお力になれることがあればお知らせください。"
            # Reset to END or INIT depending on desired flow
            session_data[user_id]["state"] = "END"
            session_data[user_id]["history"].append((user_message, response_text))
            return jsonify({
                "user_id": user_id,
                "assistant_message": response_text,
                "current_state": session_data[user_id]["state"]
            })
        else:
            # User didn't give proper feedback
            if lang == "en":
                response_text = "Please respond with 👍 or 👎."
            elif lang == "zh":
                response_text = "请回复 👍 或者 👎。"
            else:
                response_text = "👍か👎でお返事ください。"
            session_data[user_id]["history"].append((user_message, response_text))
            return jsonify({
                "user_id": user_id,
                "assistant_message": response_text,
                "current_state": session_data[user_id]["state"]
            })

    # Perform sentiment analysis
    sentiment_score = analyze_sentiment(user_message)
    session_data[user_id]["emotion_history"].append((user_message, sentiment_score))

    # Simple logic flow (For demonstration, not the full previous logic)
    # You would expand this similarly for ASK_DETAILS, HAS_INFO, SUGGEST_STRATEGY as before.
    if user_state == "INIT":
        response_text = get_response("INIT", lang, sentiment_score)
        session_data[user_id]["state"] = "END"  # Just end quickly for demo, normally you'd go to ASK_DETAILS etc.
    elif user_state == "END":
        # At the end, ask for feedback
        if lang == "en":
            response_text = get_response("END", lang, sentiment_score) + " Was this helpful? 👍 Yes / 👎 No"
        elif lang == "zh":
            response_text = get_response("END", lang, sentiment_score) + " 这次的回复对你有帮助吗？👍是的 / 👎没有"
        else:
            response_text = get_response("END", lang, sentiment_score) + " この回答は役に立ちましたか？👍 はい / 👎 いいえ"
        session_data[user_id]["state"] = "FEEDBACK"
    else:
        # Default fallback if an unknown state
        response_text = "I’m here to listen."
        session_data[user_id]["state"] = "END"

    session_data[user_id]["history"].append((user_message, response_text))

    return jsonify({
        "user_id": user_id,
        "assistant_message": response_text,
        "current_state": session_data[user_id]["state"],
        "last_sentiment_score": sentiment_score,
        "emotion_history": session_data[user_id]["emotion_history"]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)