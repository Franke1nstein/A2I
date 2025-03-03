import tensorflow as tf

try:
    loaded_model = tf.keras.models.load_model('./Model/model.h5')
    print("Simple Model loaded successfully in myAgri environment!")
    loaded_model.summary() # Optional: Print model summary to check structure
except Exception as e:
    print(f"Error loading model in myAgri environment: {e}")