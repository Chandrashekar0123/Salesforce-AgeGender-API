import numpy as np
import cv2
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import os

# Create a Flask web application instance
app = Flask(__name__)

# Define the paths to the model and cascade classifier
MODEL_PATH = 'Age_Sex_Detection.h5'
CASCADE_PATH = 'haarcascade_frontalface_default.xml'

# Load the pre-trained age and sex prediction model globally
try:
    # Using the string alias 'mae' for the custom object as determined previously
    age_sex_model = load_model(MODEL_PATH, custom_objects={'mae': 'mae'})
    print("Age_Sex_Detection.h5 loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    age_sex_model = None # Set to None if loading fails


# Load the haarcascade frontal face classifier globally
try:
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    print("haarcascade_frontalface_default.xml loaded successfully.")
except Exception as e:
    print(f"Error loading cascade classifier: {e}")
    face_cascade = None # Set to None if loading fails


# Define the preprocessing function based on previous analysis
def preprocess_image(image):
    """
    Preprocesses an image for age and sex prediction, based on the model's
    input requirements and analysis of the age-gender-identification.ipynb notebook.

    Args:
        image: The input image (NumPy array, assumed to be BGR from cv2.imread).

    Returns:
        The preprocessed image array with dimensions expanded, resized to 48x48,
        maintaining 3 channels (color), and normalized.
    """
    # Ensure the image is in color (3 channels) - cv2.imread reads as BGR by default
    if len(image.shape) == 2: # If grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Resize the image to 48x48 pixels
    resized_image = cv2.resize(image, (48, 48))

    # Convert to numpy array (already a NumPy array from cv2)
    image_array = resized_image

    # Expand dimensions to include a batch size of 1
    preprocessed_image = np.expand_dims(image_array, axis=0) # Add batch dimension


    # Normalize the image data by scaling pixel values to the range [0, 1]
    preprocessed_image = preprocessed_image / 255.0

    return preprocessed_image

# Function to make prediction, replicating the logic based on notebook analysis and raw output.
def predict_age_gender(model, image_array):
    """
    Makes age and gender predictions using the provided model and preprocessed image array,
    replicating the logic based on notebook analysis and raw output.

    Args:
        model: The loaded Keras model.
        image_array: The preprocessed image array (NumPy array with batch dimension).

    Returns:
        A tuple containing the predicted age (int), predicted gender ('Male' or 'Female'),
        and gender probability (float). Returns (None, None, None) if prediction fails.
    """
    try:
        predictions = model.predict(image_array)

        # Based on previous analysis of raw output, age is likely the second element (index 1)
        predicted_age = int(np.round(predictions[1][0])) # Age from the second output, rounded


        # Based on previous analysis of raw output, gender probability is likely the first element (index 0)
        gender_prob = predictions[0][0] # Gender probability from the first output
        predicted_gender = "Female" if gender_prob > 0.5 else "Male"
        gender_confidence = (
            gender_prob if predicted_gender == "Female" else 1 - gender_prob
        )


        return predicted_age, predicted_gender, float(gender_confidence)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None, None


@app.route('/predict_age', methods=['POST'])
def predict_age_endpoint():
    # Check if model and cascade classifier were loaded successfully
    if age_sex_model is None or face_cascade is None:
         return jsonify({'error': 'Model or cascade classifier failed to load'}), 500

    # Check if a file was uploaded in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']

    # Check if the file is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Check if the file is a JPG image (basic check based on file extension)
    if not file.filename.lower().endswith('.jpg'):
        return jsonify({'error': 'Invalid file type. Please upload a JPG image.'}), 400

    try:
        # Read the image file
        image_data = file.read()
        # Convert image data to a NumPy array
        np_image = np.frombuffer(image_data, np.uint8)
        # Decode the image from the NumPy array
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Could not decode image'}), 400

        # Convert to grayscale for face detection
        gray_image_for_detection = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray_image_for_detection, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # List to store the results for each face
        results = []

        # Process detected faces
        for (x, y, w, h) in faces:
            # Extract face region from the original color image
            face_roi = image[y:y+h, x:x+w]

            # Preprocess face using the updated function
            preprocessed_face = preprocess_image(face_roi)

            # Predict age and gender using the updated function
            predicted_age, predicted_gender, gender_confidence = predict_age_gender(age_sex_model, preprocessed_face)

            if predicted_age is not None:
                # Store only predicted age and gender in the results
                results.append({
                    'predicted_age': predicted_age,
                    'predicted_gender': predicted_gender
                })


        return jsonify({'message': 'Image processed successfully', 'detected_faces_count': len(faces), 'faces': results}), 200

    except Exception as e:
        return jsonify({'error': f'An error occurred during processing: {str(e)}'}), 500

# To run the Flask app (for local testing)
# if __name__ == '__main__':
#     app.run(debug=True)

# Note: For deployment, a WSGI server like Gunicorn is typically used.
# A Procfile for Heroku would look like: web: gunicorn app:app
# For other platforms, refer to their specific deployment documentation.

