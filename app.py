from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import base64
import cv2
import numpy as np
from deepface import DeepFace
from v import Chain  

app = Flask(__name__)
CORS(app)

emotion_chain = Chain()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400

        # Decode base64 image
        img_str = data['image'].split(',')[-1]
        img_data = base64.b64decode(img_str)
        np_img = np.frombuffer(img_data, dtype=np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Image decoding failed"}), 400

        # Resize for faster processing
        frame = cv2.resize(frame, (320, 240))

        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv'
        )

        # Sometimes DeepFace returns a list
        if isinstance(result, list):
            result = result[0]

        emotion = result.get('dominant_emotion', 'neutral')
        movie_response = emotion_chain.recommend_movies(emotion)

        return jsonify({
            "emotion": emotion,
            "recommendations": movie_response
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  