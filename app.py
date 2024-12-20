from flask import Flask, request, jsonify
from textblob import TextBlob

app = Flask(__name__)


@app.route('/sentiment', methods=['POST'])
def sentiment_analysis():
    """
    POST /sentiment
    Input:
    {
        "text": "I am happy"
    }
    Output:
    {
        "polarity": 0.8,
        "subjectivity": 0.75
    }
    Description:
    This endpoint analyzes the sentiment of the input text and returns two key metrics:
    - polarity: A float value between -1 (negative) and 1 (positive) indicating the sentiment.
    - subjectivity: A float value between 0 (objective) and 1 (subjective) indicating the subjectivity of the text.
    """
    try:
        # Extract the input text from the request
        data = request.get_json(force=True)
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing "text" field in request data'}), 400

        input_text = data.get('text', '')

        # Perform sentiment analysis using TextBlob
        blob = TextBlob(input_text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        # Return the results as JSON
        return jsonify({
            'polarity': polarity,
            'subjectivity': subjectivity
        })
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500


if __name__ == '__main__':
    # Run on localhost at port 5000
    # In production, you would run this behind a WSGI server like Gunicorn
    app.run(host='0.0.0.0', port=5000, debug=False)
