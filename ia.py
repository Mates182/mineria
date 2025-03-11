import os
import google.generativeai as genai
import markdown
import requests
from flask import Flask, request, jsonify, session, render_template
from flask_session import Session
from flask_cors import CORS  # Importar CORS para habilitar solicitudes cross-origin

app = Flask(__name__)

# Configuración de CORS para permitir solicitudes desde el frontend
CORS(app)  # Habilita CORS para todas las rutas

# Clave secreta y configuración de sesión
app.secret_key = "uce2025"
app.config["SESSION_TYPE"] = "filesystem"

Session(app)  # Inicializar sesión

# Configuración de la API de Google Generative AI
genai.configure(api_key="AIzaSyD3lYS3SCiWcG5-XimC1ipujcN-mI-ja4M")

# Cargar el modelo de IA
model = genai.GenerativeModel("gemini-1.5-flash-002")

MAX_HISTORY = 15  # Límite de mensajes en el historial
conversation_history = []


# Cargar el conocimiento base desde un archivo
def cargar_conocimiento():
    with open("static/bot.txt", "r", encoding="utf-8") as file:
        return file.read()


CONOCIMIENTO_BASE = cargar_conocimiento()


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()  # Recibir JSON del frontend
    prompt = data.get("prompt")

    if not prompt:
        return jsonify({"error": "Por favor, ingresa un texto"}), 400

    history = session.get("history", [])

    context = CONOCIMIENTO_BASE
    for item in history[-MAX_HISTORY:]:
        context += f"Usuario: {item['prompt']}\nModelo: {item['response_raw']}\n"

    conversation_history.append(f"Usuario: {prompt}")

    context = CONOCIMIENTO_BASE + "\n".join(conversation_history) + "\n"

    try:
        response = model.generate_content(context).text

        history.append(
            {
                "prompt": prompt,
                "response_raw": response,
                "response_html": markdown.markdown(response),
            }
        )
        session["history"] = history
        conversation_history.append(f"Chatbot: {response}")

        return jsonify({"response": response, "history": history}), 200

    except Exception as e:
        return jsonify({"error": f"Error al generar la respuesta: {str(e)}"}), 500


# 🔹 NUEVO ENDPOINT PARA GENERAR RECETAS ECONÓMICAS CON MENSAJE COMPLETO 🔹
@app.route("/receta", methods=["POST"])
def generar_receta():
    data = request.get_json()  # Recibir JSON del frontend
    
    # Obtener el mensaje completo
    mensaje_completo = data.get("mensaje")

    if not mensaje_completo:
        return jsonify({"error": "Debes proporcionar un mensaje completo"}), 400

    # Modificamos el prompt para utilizar el mensaje completo
    prompt = f"""Eres un asistente culinario experto en recetas económicas y deliciosas. Con base en la siguiente información: "{mensaje_completo}", sugiere una receta accesible y sabrosa.  
    Analiza el producto mencionado, su precio estimado y cualquier otro detalle relevante para elaborar una receta completa. Asegúrate de incluir los ingredientes, los pasos de preparación y consejos adicionales para mejorar el plato."""

    try:
        # Generar la receta basada en el mensaje completo
        response = model.generate_content(prompt).text
        return jsonify({"receta": response}), 200

    except Exception as e:
        return jsonify({"error": f"Error al generar la receta: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5009)
