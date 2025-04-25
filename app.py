from flask import Flask, render_template, request, redirect
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load rice grain model (update this file name if needed)
model = load_model('densenet_ieeerice_model.h5')

# 5 rice grain types
class_names = ['Polished Rice', 'Basmathi', 'Sona Masuri', 'Nellore', 'Brown Rice']

# Ensure upload folder exists
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        img_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(img_path)

        # Preprocess image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Predict
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]

        return render_template('index.html', prediction=predicted_class, image_url=img_path)

if __name__ == '__main__':
    app.run(debug=True)
