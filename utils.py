import numpy as np
import tensorflow as tf
import json

def process_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img /= 255.0
    return img.numpy()

def load_category_names(category_names_path):
    with open(category_names_path, 'r') as f:
        return json.load(f)