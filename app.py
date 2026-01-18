import os
import sys
import subprocess
import numpy as np
import tensorflow as tf
def install(package):
    print(f"Instaling library: {package}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"{package} instaled.")
    except Exception as e:
        print(f"Error instaling {package}: {e}")
try:
    import google.generativeai as genai
except ImportError:
    install("google-generativeai")
    import google.generativeai as genai

try:
    from dotenv import load_dotenv
except ImportError:
    install("python-dotenv")
    from dotenv import load_dotenv

carpeta_actual = os.path.dirname(os.path.abspath(__file__))
ruta_env = os.path.join(carpeta_actual, '.env')
load_dotenv(ruta_env)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("ERROR: api key not found in .env")
else:
    print(f"Key loaded succesfully: {GOOGLE_API_KEY[:5]}...")
    genai.configure(api_key=GOOGLE_API_KEY)



from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import io
import base64
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

#print("loading model... please wait....")
#try:
#    model = tf.keras.models.load_model('model_skin.h5')
 #   print("Model loaded successfully.")
#except Exception as e:
  #  print(f"Error loading model: {e}")
  #  print("Make sure the model file 'model_skin.h5' is in the correct directory.")

model = None  # 1. Definimos la variable vacía para evitar el NameError

print("🔍 INICIANDO DIAGNÓSTICO DE CARGA...")
print(f"📂 Directorio actual de trabajo: {os.getcwd()}")
print(f"📄 Archivos en este directorio: {os.listdir('.')}")

try:
    # 2. Asegúrate de que este nombre sea IDÉNTICO al
    model_path = 'model_skin.h5'

    if os.path.exists(model_path):
        print(f"✅ El archivo {model_path} EXISTE. Intentando cargar...")
        model = load_model(model_path)
        print("🎉 MODELO CARGADO EXITOSAMENTE en memoria.")
    else:
        print(f"❌ ERROR FATAL: No encuentro el archivo '{model_path}'.")
        print("   ¿Está en una subcarpeta? ¿Está mal escrito el nombre?")

except Exception as e:
    print(f"🔥 ERROR CRÍTICO DE TENSORFLOW: {e}")

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
        model_gemini = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"""
        You are an Expert Dermatologist AI.
        
        INPUT:
        - CNN Prediction: "{diagnosis}" ({confidence:.1f}%).
        
        INTERNAL VISUAL ANALYSIS (DO NOT OUTPUT THIS PART):
        1. Compare "Red Blobs" (Vascular) vs "Red Branching Lines" (BCC).
        2. Look for "Central White Patch" (Dermatofibroma).
        
        DECISION LOGIC:
        - IF you see branching red lines (arborizing vessels) + shiny skin -> DIAGNOSIS: Basal Cell Carcinoma (BCC).
        - IF you see round red/purple clumps (lacunae) -> DIAGNOSIS: Vascular Lesion.
        - IF you see a central white scar-like patch -> DIAGNOSIS: Dermatofibroma.
        - OTHERWISE -> Support the CNN prediction.

        OUTPUT INSTRUCTIONS:
        - ONLY output the final HTML result. 
        - DO NOT list the steps, checklists, or internal thinking.
        
        REQUIRED HTML FORMAT:
        <b>AI Second Opinion:</b> [Your Verdict]
        <br>
        <b>Visual Evidence:</b> [One clear sentence describing the features you see]
        <br>
        <b>Recommendation:</b> [Short medical advice]
        """

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