from flask import Flask, request, jsonify
import cv2
import numpy as np
from keras_facenet import FaceNet
import joblib
from flask_cors import CORS


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all origins

# Load models
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
embedder = FaceNet()
loaded_model = joblib.load("cnn_model.pkl")
loaded_label_encoder = joblib.load("label_encoder.pkl")

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.1, 4)
    return faces

def extract_embeddings(face_images):
    embeddings = []
    for face_image in face_images:
        resized_image = cv2.resize(face_image, (160, 160))
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        input_image = np.expand_dims(resized_image, axis=0)
        embedding = embedder.embeddings(input_image)
        embeddings.append(embedding)
    return embeddings

def predict(image):
    # Predict labels for the image
    faces = detect_faces(image)
    face_images = [image[y:y+h, x:x+w] for (x, y, w, h) in faces]
    embeddings = extract_embeddings(face_images)
    predicted_labels = [loaded_model.predict(embedding) for embedding in embeddings]
    return predicted_labels

@app.route('/api', methods=['GET'])
def return_ascii():
    d = {}
    input_chr = str(request.args['query'])
    answer = str(ord(input_chr))
    d['output'] = answer
    d['result'] = "21bcs052"
    return jsonify(d)

@app.route('/', methods=["GET"])
def hello():
    d = {}
    d['pred'] = "test"
    return jsonify(d)

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        # Check if image file is present in the request
        if 'image' not in request.files:
            print("error!!")
            return jsonify({'error': 'No image provided'}), 400
        
        # Read image file from request
        image_file = request.files['image']
        image_np = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        # Predict labels for the image
        prediction = predict(image)

        # Format and return prediction
        labels = []
        for i in prediction:
            print(i)
            predicted_label_index = np.argmax(i)
            if predicted_label_index <= 10 and i[0][predicted_label_index]>=0.95:
                predicted_label = loaded_label_encoder.inverse_transform([predicted_label_index])[0]
            else:
                predicted_label = "UNKNOWN"
            labels.append(predicted_label)
        unique_labels = list(set(labels))
        print(unique_labels)
        return jsonify({'predicted_labels': unique_labels}), 200
    
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='10.0.2.250', debug=True, use_reloader = False)
