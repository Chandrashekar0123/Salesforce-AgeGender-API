import numpy as np
import cv2
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import os

# Flask application
app = Flask(__name__)

# Paths to model and Haar cascade
MODEL_PATH = 'Age_Sex_Detection.h5'
CASCADE_PATH = 'haarcascade_frontalface_default.xml'

# Load model
try:
    age_sex_model = load_model(MODEL_PATH, custom_objects={'mae': 'mae'})
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    age_sex_model = None

# Load Haar cascade
try:
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        raise Exception("Cascade file is empty or invalid.")
    print("✅ Haarcascade loaded successfully.")
except Exception as e:
    print(f"❌ Error loading Haarcascade: {e}")
    face_cascade = None

# Preprocessing function
def preprocess_image(image):
    if len(image.shape) == 2:  # grayscale → BGR
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    resized_image = cv2.resize(image, (48, 48))
    preprocessed_image = np.expand_dims(resized_image, axis=0) / 255.0
    return preprocessed_image

# Prediction function
def predict_age_gender(model, image_array):
    try:
        predictions = model.predict(image_array)
        predicted_age = int(np.round(predictions[1][0]))
        gender_prob = predictions[0][0]
        predicted_gender = "Female" if gender_prob > 0.5 else "Male"
        gender_confidence = gender_prob if predicted_gender == "Female" else 1 - gender_prob
        return predicted_age, predicted_gender, float(gender_confidence)
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None, None

@app.route('/')
def index():
    return jsonify({"message": "Age & Gender Prediction API is running."})

@app.route('/predict_age', methods=['POST'])
def predict_age_endpoint():
    if age_sex_model is None or face_cascade is None:
        return jsonify({'error': 'Model or cascade not loaded properly'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        return jsonify({'error': 'Invalid file type. Use JPG, JPEG or PNG'}), 400

    try:
        image_data = file.read()
        np_image = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Could not decode image'}), 400

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        results = []
        for (x, y, w, h) in faces:
            face_roi = image[y:y+h, x:x+w]
            preprocessed_face = preprocess_image(face_roi)
            predicted_age, predicted_gender, _ = predict_age_gender(age_sex_model, preprocessed_face)
            if predicted_age is not None:
                results.append({
                    'predicted_age': predicted_age,
                    'predicted_gender': predicted_gender
                })

        return jsonify({
            'message': 'Image processed successfully',
            'detected_faces_count': len(faces),
            'faces': results
        }), 200

    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

# For local testing
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
