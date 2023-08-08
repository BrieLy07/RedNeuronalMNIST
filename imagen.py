import numpy as np
from PIL import Image
import requests
from io import BytesIO

# Cargar una imagen desde una URL
image_url = "https://interactivechaos.com/sites/default/files/2020-09/tutdl_0072.jpg" 
 # Reemplaza con la URL de la imagen que deseas probar

response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

# Redimensionar la imagen a 28x28 píxeles (tamaño de entrada del modelo)
image = image.resize((28, 28))
image = np.array(image)
image = image[:, :, 0]  # Si la imagen es en color, convertirla a blanco y negro

# Preprocesar la imagen
image = image / 255.0
image = np.reshape(image, (1, 28, 28))  # Cambiar la forma a (1, 28, 28) para que sea una matriz de entrada

# Cargar el modelo previamente entrenado (asegúrate de ajustar la ruta al modelo)
from keras.models import load_model
model = load_model("C:/Users/gabri/Desktop/UNIVERSIDAD/UCE/SEXTO/ANALISIS DE DATOS/EXPOSICION/EXPO/modelo_entrenado.h5")

# Hacer la predicción en la imagen
prediction = model.predict(image)
predicted_class = np.argmax(prediction)

# Mostrar los resultados
print("Clase predicha:", predicted_class)
