from flask import Flask, request, render_template
import numpy as np
from PIL import Image
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

(_, _), (X_val, y_val) = mnist.load_data()
X_val = X_val.reshape(X_val.shape[0], 28, 28, 1)
X_val = X_val / 255.0

# Get the directory where this script resides
script_dir = os.path.dirname(os.path.realpath(__file__))

# Load saved models using relative paths
cnn_model = load_model(os.path.join(script_dir, 'cnn_model'))
ann_model = load_model(os.path.join(script_dir, 'ann_model'))
rnn_model = load_model(os.path.join(script_dir, 'rnn_model'))

# Define function to preprocess image
def preprocess_image(img_path):
    img = Image.open(img_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = img_array.reshape(1, 28, 28, 1) / 255.0
    return img_array

# Define function to evaluate models on validation dataset
def evaluate_model(model):
    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
    return accuracy

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['file']

        # If the user does not select a file, the browser submits an empty file without filename
        if file.filename == '':
            return render_template('index.html', error='No selected file')

        if file:
            # Preprocess the image
            img_array = preprocess_image(file)

            # Make predictions using each model
            cnn_predictions = cnn_model.predict(img_array)
            cnn_predicted_class = int(np.argmax(cnn_predictions))
            cnn_validation_accuracy = evaluate_model(cnn_model)

            ann_predictions = ann_model.predict(img_array)
            ann_predicted_class = int(np.argmax(ann_predictions))
            ann_validation_accuracy = evaluate_model(ann_model)

            rnn_predictions = rnn_model.predict(img_array)
            rnn_predicted_class = int(np.argmax(rnn_predictions))
            rnn_validation_accuracy = evaluate_model(rnn_model)

            # Determine the best model
            accuracies = {'CNN': cnn_validation_accuracy, 'ANN': ann_validation_accuracy, 'RNN': rnn_validation_accuracy}
            best_model = max(accuracies, key=accuracies.get)

            return render_template('result.html',
                                   cnn_predicted_class=cnn_predicted_class, cnn_accuracy=cnn_validation_accuracy,
                                   ann_predicted_class=ann_predicted_class, ann_accuracy=ann_validation_accuracy,
                                   rnn_predicted_class=rnn_predicted_class, rnn_accuracy=rnn_validation_accuracy,
                                   best_model=best_model)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
