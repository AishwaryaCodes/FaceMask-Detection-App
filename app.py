from flask import Flask, render_template, request, jsonify
from tensorflow import keras
import numpy as np
import cv2
import base64

app = Flask(__name__)

# Load the face mask detector model
maskNet = keras.models.load_model("mask_detector.h5")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/detect-mask", methods=["POST"])
def detect_mask():
    try:
        # Get Base64 image from request
        data = request.json
        if 'image' not in data:
            return jsonify({"error": "No image provided!"}), 400

        image_data = data['image'].split(",")[1]  # Remove the data:image/jpeg;base64, part
        image_bytes = base64.b64decode(image_data)

        # Convert to numpy array and preprocess
        np_image = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (224, 224))
        image = keras.preprocessing.image.img_to_array(image)
        image = keras.applications.mobilenet_v2.preprocess_input(image)
        image = np.expand_dims(image, axis=0)

        # Make prediction
        (mask, withoutMask) = maskNet.predict(image)[0]
        label = "Mask" if mask > withoutMask else "No Mask"
        confidence = max(mask, withoutMask) * 100

        return jsonify({"label": label, "confidence": f"{confidence:.2f}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
