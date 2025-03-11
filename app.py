from flask import Flask, render_template, request, jsonify, flash
import joblib
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import losses

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Necesario para mostrar mensajes de error

# Definir la métrica manualmente
def mse(y_true, y_pred):
    return losses.mean_squared_error(y_true, y_pred)

# Cargar el modelo con la métrica correcta registrada
modelo = tf.keras.models.load_model('modelo/modelo_precio_mercados.h5', custom_objects={'mse': mse})

scaler = joblib.load('modelo/scaler.pkl')
label_encoders = joblib.load('modelo/label_encoders.pkl')

predictoras = ['Año', 'Mes', 'Día', 'Provincia', 'Cantón', 'Producto', 'Pres.', 'Cant.', 'Unidad Medida']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()

        # Validaciones
        for key in ['Año', 'Mes', 'Día', 'Cant.']:
            if not data[key].isdigit():
                flash(f"⚠️ {key} debe ser un número válido.", "error")
                return render_template('index.html')

        data['Año'] = int(data['Año'])
        data['Mes'] = int(data['Mes'])
        data['Día'] = int(data['Día'])
        data['Cant.'] = float(data['Cant.'])

        if not (1 <= data['Mes'] <= 12):
            flash("⚠️ El mes debe estar entre 1 y 12.", "error")
            return render_template('index.html')

        if not (1 <= data['Día'] <= 31):
            flash("⚠️ El día debe estar entre 1 y 31.", "error")
            return render_template('index.html')

        # Codificación de variables categóricas
        for col in ['Provincia', 'Cantón', 'Producto', 'Pres.', 'Unidad Medida']:
            if data[col] not in label_encoders[col].classes_:
                flash(f"⚠️ {col} no es un valor válido.", "error")
                return render_template('index.html')

            data[col] = label_encoders[col].transform([data[col]])[0]

        # Crear el array para la predicción
        input_data = np.array([[data['Año'], data['Mes'], data['Día'], data['Provincia'], data['Cantón'],
                                data['Producto'], data['Pres.'], data['Cant.'], data['Unidad Medida']]])

        input_data = scaler.transform(input_data)

        prediccion = modelo.predict(input_data).flatten()[0]

        # Decodificar el nombre del producto
        producto_codificado = data['Producto']
        producto = label_encoders['Producto'].inverse_transform([producto_codificado])[0]

        # Mostrar el mensaje con el nombre del producto y el precio estimado
        flash(f"✅ El precio estimado para el producto '{producto}' es: {round(prediccion, 2)} USD/Kg", "success")
        
        return render_template('index.html')

    except Exception as e:
        flash(f"❌ Error en la predicción: {str(e)}", "error")
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5008)