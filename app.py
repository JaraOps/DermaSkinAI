import os
import numpy as np
import tensorflow as tf
try:
    import google.generativeai as genai
except ImportError:
    genai = None
    print("google-generativeai no está instalado. Gemini deshabilitado.")
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import io
import base64
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from dotenv import load_dotenv

app = Flask(__name__)
GOOGLE_API_KEY= os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("No GOOGLE_API_KEY env variable.")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

print("loading model... please wait....")
try:
    model = tf.keras.models.load_model('model_skin.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure the model file 'model_skin.h5' is in the correct directory.")

CLASES = {
    0: 'Actinic Keratosis (akiec)',
    1: 'Basal Cell Carcinoma (bcc)',
    2: 'Benign Keratosis (bkl)',
    3: 'Dermatofibroma (df)',
    4: 'Melanoma (mel)',
    5: 'Melanocytic Nevus (nv)',
    6: 'Vascular Lesion (vasc)'
}

def prepare_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((224, 224))
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def ask_gemini(diagnosis, confidence, image):
    try:
        model_gemini = "gemini-1.5-flash"
        prompt = f"""
        Act as a professional dermatology assistant .

        Context: A Deep Learning model (MobileNetV2) analyzed this image and predicted: 
        "{diagnosis}" with {confidence:.1f}% confidence.

        Your task:
        
        1. Explain in 1 simple sentence what "{diagnosis}" is.
        2. Visually analyze the attached image based on the ABCD rule (Asymmetry, Border, Color, Diameter).
        3. Provide a clear recommendation (e.g., visit a specialist).

        Format your response in simple HTML (use <b> for bold, <ul><li> for lists).
        IMPORTANT: Start with a disclaimer that this is an AI prototype, not a doctor."""

        response = model_gemini.generate_content([prompt, image])
        return response.text
    except Exception as e:
        return f"Error generating Gemini response: {e}"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            image = Image.open(file.stream)
            processed_img = prepare_image(image)
            preds = model.predict(processed_img)
            pred_index = np.argmax(preds)
            confidence = np.max(preds)*100
            diagnosis = CLASES.get(pred_index, "Unknown")
            explanation = ask_gemini(diagnosis, confidence, image)
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            encoded_img = base64.b64encode(img_byte_arr.getvalue()).decode('ascii')
        return render_template('index.html',
                                   prediction = diagnosis,
                                   confidence = f"{confidence:.2f}",
                                   explanation = explanation,
                                   img_data = encoded_img)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)