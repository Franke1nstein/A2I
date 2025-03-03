import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

#Tensorflow Model Prediction
def model_prediction(test_image, model_path):
    model = tf.keras.models.load_model(model_path)
    # Resize to 256x256 to match expected input shape of the model
    img = image.load_img(test_image, target_size=(256, 256))
    input_arr = image.img_to_array(img)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    # Retourner à la fois l'index de la classe prédite ET les probabilités
    predicted_class_index = np.argmax(predictions)
    return predicted_class_index, predictions