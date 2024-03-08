import os
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from PIL import Image
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import cv2
import numpy as np
import mysql.connector
import base64

app = Flask(__name__)
app.secret_key = '123'  # Change this to a random secret key

# Load the trained models
BREED_MODEL_PATH = 'breedCheckpoint\\epoch_5_checkpoint(12).pth'
HEALTH_MODEL_PATH = 'healthCheckpoint\\epoch_5_checkpoint(13).pth'
EMOTION_MODEL_PATH = 'emotionPrediction\\checkpoints\\epoch_5_checkpoint(14).pth'
AGE_MODEL_PATH = 'agePrediction\\epoch_5_checkpoint(15).pth'
GENDER_MODEL_PATH = 'genderPrediction\\inception_epoch_5(5).pth'

breed_model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=10)
health_model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=3)
emotion_model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=4)
age_model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=3)
gender_model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=False)
gender_model.fc = torch.nn.Linear(gender_model.fc.in_features, 2)  # Assuming 2 classes for gender
gender_model.load_state_dict(torch.load(GENDER_MODEL_PATH, map_location=torch.device('cpu')))

breed_model.load_state_dict(torch.load(BREED_MODEL_PATH, map_location=torch.device('cpu')))
health_model.load_state_dict(torch.load(HEALTH_MODEL_PATH, map_location=torch.device('cpu')))
emotion_model.load_state_dict(torch.load(EMOTION_MODEL_PATH, map_location=torch.device('cpu')))
age_model.load_state_dict(torch.load(AGE_MODEL_PATH, map_location=torch.device('cpu')))

breed_model.eval()
health_model.eval()
emotion_model.eval()
age_model.eval()
gender_model.eval()

# Transformations
transform_new_image = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Class mappings
class_to_breed = {
    0: 'Abyssinian',
    1: 'American Shorthair',
    2: 'Balinese',
    3: 'Bengal',
    4: 'Birman',
    5: 'Bombay',
    6: 'British Shorthair',
    7: 'Persian',
    8: 'Siamese',
    9: 'Sphynx'
}

class_to_health = {
    0: 'have good stamina and can maintain its activity levels.',
    1: 'might not be the most energetic due to their weight',
    2: 'might need some time to bounce back to full stamina'
}

class_to_emotion = {
    0: 'Angry',
    1: 'Happy',
    2: 'Relaxed',
    3: 'Sad'
}

class_to_age = {
    0: 'Kitten - Age Range : 0-1 year old',
    1: 'Young Cat - Age Range : 1-10 years old',
    2: 'Old Cat - Age Range : 10+ years old'
}

class_to_gender = {
    0: 'Female',
    1: 'Male'
}

def predict_with_confidence(model, image, class_mapping, top_n=3):
    image = transform_new_image(image)
    image = image.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        top_probs, top_classes = torch.topk(probabilities, min(top_n, len(class_mapping)))
        results = [(class_mapping[class_id.item()], prob.item()) for class_id, prob in zip(top_classes[0], top_probs[0])]
    return results

# Ensure you have the MySQL server running and the specified database and user exist.
mydb = mysql.connector.connect(
    host="localhost",
    port="3307",  # Default MySQL port
    user="root",
    password="",  # Use your MySQL root password
    database="animal_data"
)

mycursor = mydb.cursor(buffered=True)

mycursor.execute("CREATE TABLE IF NOT EXISTS cats (id INT AUTO_INCREMENT PRIMARY KEY, image LONGBLOB, breed VARCHAR(100), gender VARCHAR(10), age VARCHAR(10), stamina VARCHAR(50), emotion VARCHAR(50))")

# Admin password
ADMIN_PASSWORD = "123"

def is_admin(request):
    return request.form.get("admin_password") == ADMIN_PASSWORD

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if is_admin(request):
            if 'add_cat' in request.form:
                image_file = request.files['image']
                image_data = image_file.read()
                breed = request.form['breed']
                gender = request.form['gender']
                age = request.form['age']
                stamina = request.form['stamina']
                emotion = request.form['emotion']

                sql = "INSERT INTO cats (image, breed, gender, age, stamina, emotion) VALUES (%s, %s, %s, %s, %s, %s)"
                val = (image_data, breed, gender, age, stamina, emotion)
                mycursor.execute(sql, val)
                mydb.commit()
                flash('Cat added successfully!', 'success')
                return redirect(url_for('index'))

            elif 'edit_cat' in request.form:
                cat_id = request.form['cat_id']
                breed = request.form['edit_breed']
                gender = request.form['edit_gender']
                age = request.form['edit_age']
                stamina = request.form['edit_stamina']
                emotion = request.form['edit_emotion']

                sql = "UPDATE cats SET breed = %s, gender = %s, age = %s, stamina = %s, emotion = %s WHERE id = %s"
                val = (breed, gender, age, stamina, emotion, cat_id)
                mycursor.execute(sql, val)
                mydb.commit()
                flash('Cat updated successfully!', 'success')
                return redirect(url_for('index'))

            elif 'delete_cat' in request.form:
                cat_id = request.form['cat_id']
                sql = "DELETE FROM cats WHERE id = %s"
                val = (cat_id,)
                mycursor.execute(sql, val)
                mydb.commit()
                flash('Cat deleted successfully!', 'success')
                return redirect(url_for('index'))
        else:
            flash('Unauthorized access! Please enter admin password.', 'error')
            return redirect(url_for('index'))

    mycursor.execute("SELECT * FROM cats")
    cats = mycursor.fetchall()
    cat_records = []
    for cat in cats:
        cat_record = list(cat)
        cat_image_encoded = base64.b64encode(cat[1]).decode('utf-8')
        cat_record[1] = f"data:image/jpeg;base64,{cat_image_encoded}"
        cat_records.append(cat_record)
    return render_template('index15.html', cats=cat_records)

@app.route('/predict', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image sent.'}), 400
    image_file = request.files['image']
    try:
        image = Image.open(image_file)
        image_np = np.array(image)  # Convert PIL Image to numpy array
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # Only if needed
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

        cat_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalcatface.xml')
        faces = cat_cascade.detectMultiScale(gray, 1.1, 3)
        num_faces_detected=len(faces)
        if num_faces_detected == 0:
            return jsonify({'error': 'No cat face detected in the image.'}), 400

        breed_predictions = predict_with_confidence(breed_model, image, class_to_breed, top_n=3) # changed to top_n=3
        breed_results = {f"breed_{i+1}": {"name": pred[0], "confidence": f"{pred[1]*100:.2f}%"} for i, pred in enumerate(breed_predictions)}
        health, health_confidence = predict_with_confidence(health_model, image, class_to_health)[0]
        emotion, emotion_confidence = predict_with_confidence(emotion_model, image, class_to_emotion)[0]
        age, age_confidence = predict_with_confidence(age_model, image, class_to_age)[0]
        gender, gender_confidence = predict_with_confidence(gender_model, image, class_to_gender)[0]

        return jsonify({**breed_results, 'num_faces': num_faces_detected, 'cat_detected': True,
                        'health': health, 'health_confidence': health_confidence,
                        'emotion': emotion, 'emotion_confidence': emotion_confidence,
                        'age': age, 'age_confidence': age_confidence,
                        'gender': gender, 'gender_confidence': gender_confidence}), 200
    except Exception as e:
        return jsonify({'error': 'Error processing image.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
