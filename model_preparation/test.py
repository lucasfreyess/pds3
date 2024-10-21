import cv2
import numpy as np
import tensorflow as tf

# Cargar el modelo .h5
model = tf.keras.models.load_model('models/benjita.keras')

# list_of_gestures = ['1', "  10", '2', '3', '4', '5', "Blank"] # Isaías
# list_of_gestures = ['A', 'B', "Three", "Five", "One", "Two", "Blank"] # mías v2
# list_of_gestures = ['A', 'B', "One", "Two", "Three", "Five", "Blank"] # mías v3
list_of_gestures = ['5', '0', '1', "2", "3", "4"] # mías v3

# Definir el tamaño de las imágenes que el modelo espera
IMG_WIDTH, IMG_HEIGHT = 96, 96

# Iniciar captura de video desde la cámara
cap = cv2.VideoCapture(0)

while True:
    # Leer un frame del video
    ret, frame = cap.read()
    
    if not ret:
        print("Error al capturar el video.")
        break

    # Invertir horizontalmente el frame
    frame = cv2.flip(frame, 1)  # 1 para inversión horizontal

    # Redimensionar la imagen al tamaño esperado por el modelo
    input_image = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    
    # Normalizar los valores de los píxeles entre 0 y 1
    input_image = input_image / 255.0

    # Convertir a tipo uint8 antes de pasar a escala de grises
    input_image = (input_image * 255).astype(np.uint8)
    
    # Convertir la imagen a escala de grises
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('Imagen.jpg', input_image)

    # Aplanar la imagen para que tenga la forma correcta (1, IMG_WIDTH, IMG_HEIGHT, 1)
    input_image = input_image.reshape(1, IMG_WIDTH, IMG_HEIGHT, 1)
    
    # Hacer la predicción
    prediction = model.predict(input_image)
    print(f"Clase predicha: {prediction}")
    
    # Obtener la clase con mayor probabilidad
    predicted_class = np.argmax(prediction, axis=1)
    
    # Mostrar la clase predicha en el frame del video
    cv2.putText(frame, f"Prediction: {list_of_gestures[int(predicted_class[0])]} ({predicted_class[0]})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Mostrar el video con la predicción
    cv2.imshow('Video en tiempo real', frame)

    # Salir del loop si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()
