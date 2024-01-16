import base64
import numpy as np
import io
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from flask import request, jsonify, abort, Flask
import logging
import os

# Configure logging
logging.basicConfig(filename='app.log', level=logging.DEBUG)

MODEL_FILE = os.environ.get('MODEL_FILE', 'MV2_RMS_best_model.h5')
app = Flask(__name__)

def get_model(model_file):
    model = load_model(model_file)
    logging.info("Model loaded!")
    return model

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

print(" * Loading Keras model...")

# Load the model
model = get_model(MODEL_FILE)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        message = request.get_json(force=True)
        encoded = message.get('image')

        if not encoded:
            abort(400, "Bad Request: 'image' key not found in the request payload.")

        decoded = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(decoded))

        if image.format.lower() not in ['jpeg', 'png', 'gif']:
            abort(400, "Bad Request: Invalid image format. Supported formats: JPEG, PNG, GIF.")

        processed_image = preprocess_image(image, target_size=(224, 224))

        # Get model prediction
        predicted_probs = model.predict(processed_image)
        predicted_label = int(predicted_probs > 0.5)

        response = {
            'prediction': {
                'defective_probability': float(predicted_probs[0][0]),
                'defective_label': predicted_label,
            }
        }
        logging.info("Prediction successful.")
        return jsonify(response)

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        abort(500, "Internal Server Error")

if __name__ == '__main__':
    app.run(debug=False)
