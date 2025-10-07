

<div align="center">
  <h1>Salesforce Age & Gender Detection API</h1>
  <b>AI-powered API for real-time age and gender prediction from images, built for Salesforce integration.</b>
</div>

---

## 🚀 Overview
This project is a hackathon solution for Cognizant, providing a RESTful API to detect age and gender from face images using deep learning. Easily deployable and integrable with Salesforce and other platforms.

---

## 🧩 Features
- Predicts age and gender from uploaded face images
- Fast, accurate deep learning model (Keras/TensorFlow)
- REST API
- Ready for deployment (Render/Cloud)
- Salesforce integration-ready
- Face detection using Haarcascade

---

## 🛠️ Technology Stack
- Python 3.9
- TensorFlow / Keras
- OpenCV
- Render (for deployment)
- Haarcascade (face detection)

---

## 📦 Setup & Installation
```bash
git clone https://github.com/Chandrashekar0123/Salesforce-AgeGender-API.git
cd Salesforce-AgeGender-API
pip install -r requirements.txt
python app.py
```

---

## 🖼️ Example Request
Send a POST request to `/predict_age` with an image file:
```bash
curl -X POST -F "image=@your_image.jpg" http://localhost:5000/predict_age
```

---

<img width="1280" height="612" alt="image" src="https://github.com/user-attachments/assets/cc13d590-5141-4709-bb56-53ca829d09b9" />


---

## 📊 Architecture
![Architecture Diagram](Architecture%20Diagram.jpg)

---

## 💡 How it Works
1. **Face Detection:** Uses Haarcascade to detect faces in the image.
2. **Preprocessing:** Crops and resizes detected faces.
3. **Prediction:** Deep learning model predicts age and gender.
4. **Response:** Returns JSON with age and gender for each detected face.

---

## 🌐 Deployment
- Ready for Render: Includes `Procfile` and `runtime.txt`
- Gunicorn for production server

---

## 📚 Documentation
- See [Document.pdf](Document.pdf) and [PPT.pdf](PPT.pdf) for detailed explanation and presentation.

---

## 🏆 Hackathon Project
Developed for Cognizant Hackathon 2025.

---

## 👤 Author
**Chandrashekar0123**

---


<div align="center">
	<img src="Technology%20Stack.jpg" width="600" style="margin: 20px;"/>
	<br>
	<img src="https://github.com/user-attachments/assets/599b8c0d-8313-45e2-ab7a-005b74fdd2e6" width="600" style="margin: 20px;"/>
</div>
