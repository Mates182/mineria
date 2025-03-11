import os
import requests
from flask import Flask, jsonify, request, render_template, send_file
from flask_cors import CORS  # Importar CORS para habilitar solicitudes cross-origin
from access import tokenH  # Asegúrate de tener el token de Hugging Face en access.py

app = Flask(__name__)
CORS(app)  # Habilita CORS para todas las rutas

# URL del modelo en Hugging Face (Stable Diffusion)
url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-3.5-large"
token = tokenH  # Reemplaza con tu token de Hugging Face

@app.route('/')
def index():
    return render_template('image.html')

@app.route('/generate-image', methods=['POST'])
def generate_image():
    # Obtener el prompt desde la solicitud
    prompt = request.form.get('prompt', 'Plato con carne')  # Usamos un valor por defecto

    # Datos para la solicitud
    request_data = {
        "inputs": prompt,
        "options": {"use_gpu": True},
    }

    headers = {
        "Authorization": f"Bearer {token}"
    }

    response = requests.post(url, headers=headers, json=request_data)

    if response.status_code == 200:
        # Guardar la imagen generada
        image_data = response.content
        image_path = "static/generated_image.png"  # Guardarla en la carpeta estática
        
        with open(image_path, "wb") as image_file:
            image_file.write(image_data)
        
        # Devolver la URL de la imagen generada
        return jsonify({"image_url": f"/static/generated_image.png"})
    else:
        return jsonify({"error": f"Error al generar la imagen: {response.text}"}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5010)
