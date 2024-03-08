import os
from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import cv2
import numpy as np

app = Flask(__name__)

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
    #0: 'Healthy',
    #1: 'Overweight',
    #2: 'Skin Disease'
    0:'have good stamina and can maintain its activity levels.',
    1:'might not be the most energetic due to their weight',
    2:'might need some time to bounce back to full stamina'
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


@app.route('/')
def index():
    return render_template('index14.html')

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
