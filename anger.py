from flask import Flask, request, jsonify

app = Flask(__name__)

# Simplified anger word dictionaries
anger_dictionaries = {
    'en': {'angry', 'mad', 'furious', 'rage', 'irritated', 'annoyed'},
    'zh': {'生气', '愤怒', '怒火', '暴怒', '恼火'},
    'ja': {'怒り', 'イライラ', '憤怒', '激怒'}
}

def validate_language(language: str) -> bool:
    """Check if the requested language is supported."""
    return language in anger_dictionaries

def analyze_text(input_text: str, language: str) -> dict:
    """Analyze the text for anger intensity."""
    anger_words = anger_dictionaries[language]
    # Split by whitespace; for more sophisticated tokenization, consider regex or NLP libraries.
    words_in_text = input_text.split()
    matching_words = [word for word in words_in_text if word.lower() in anger_words]
    total_words = len(words_in_text)
    intensity = len(matching_words) / total_words if total_words > 0 else 0
    return {
        "total_words": total_words,
        "matching_words": matching_words,
        "intensity": round(intensity, 4)
    }

@app.route('/detect_anger', methods=['POST'])
def detect_anger():
    """Detects the intensity of anger in a given text."""
    data = request.get_json()
    if data is None:
        return jsonify({'error': 'Invalid JSON'}), 400

    input_text = data.get('text', '').strip()
    if not input_text:
        return jsonify({'error': 'The "text" field is required and cannot be empty'}), 400

    language = data.get('language', 'en').lower()
    if not validate_language(language):
        return jsonify({
            'error': f"Unsupported language: {language}. Supported languages are {list(anger_dictionaries.keys())}"
        }), 400

    # Validate confidence threshold
    try:
        confidence_threshold = float(data.get('confidence_threshold', 0.5))
        if not 0 <= confidence_threshold <= 1:
            raise ValueError()
    except ValueError:
        return jsonify({'error': 'The "confidence_threshold" must be a number between 0 and 1'}), 400

    analysis_result = analyze_text(input_text, language)
    anger_detected = analysis_result["intensity"] >= confidence_threshold

    return jsonify({
        'emotion': 'anger',
        'language': language,
        'input_text': input_text,
        'total_words': analysis_result["total_words"],
        'matching_words': len(analysis_result["matching_words"]),
        'matching_word_list': analysis_result["matching_words"],
        'intensity': analysis_result["intensity"],
        'confidence_threshold': confidence_threshold,
        'anger_detected': anger_detected
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
