from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

app = Flask(__name__)

model = tf.keras.models.load_model(r"C:\Users\Admin\Desktop\firsttest_cnn_website\cnn_number_pred.h5")

UPLOAD_FOLDER = r'C:\Users\Admin\Desktop\firsttest_cnn_website\upload_folder'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def model_predict(imgpath, model):
    img = cv2.imread(imgpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28))
    img = img.reshape(28, 28, 1).astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    return np.argmax(predictions)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            result = model_predict(filepath, model)
            return render_template("result.html", prediction=result)

    return render_template("index.html")
    
if __name__ == "__main__":
    app.run(debug=True)
