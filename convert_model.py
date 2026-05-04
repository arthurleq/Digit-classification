# convert_model.py
import tensorflow as tf

model = tf.keras.models.load_model("model.h5")
model.export("saved_model")  # Create a model in the SavedModel format