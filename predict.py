import argparse
import numpy as np
import tensorflow as tf
from utils import process_image, load_category_names

def predict(image_path, model_path, top_k, category_names_path):
    model = tf.keras.models.load_model(model_path)
    img = process_image(image_path)
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    probs, classes = tf.math.top_k(predictions, k=top_k)
    probs = probs.numpy().squeeze()
    classes = classes.numpy().squeeze()
    if category_names_path:
        category_names = load_category_names(category_names_path)
        class_names = [category_names[str(c+1)] for c in classes]
    else:
        class_names = [str(c+1) for c in classes]
    for i in range(top_k):
        print(f"Class {class_names[i]} with probability {probs[i]:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict flower class from an image')
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('model_path', help='Path to the saved Keras model')
    parser.add_argument('--top_k', type=int, default=5, help='Return the top K most likely classes')
    parser.add_argument('--category_names', help='Path to a JSON file mapping labels to flower names')
    args = parser.parse_args()
    predict(args.image_path, args.model_path, args.top_k, args.category_names)
