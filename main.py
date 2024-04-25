import numpy as np
from joblib import load
from flask import Flask, jsonify, request
import csv
from paddleocr import PaddleOCR
import cv2
from PIL import Image
import fitz
from io import BytesIO
import base64
import requests

app = Flask(__name__)

allowed_extentsions = {'pdf', 'jpg', 'jpeg'}

# Load OCR model
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Load diabetes prediction model
diabetes_model = load('Models/diabetes_model.joblib')

# Load liver disease prediction model
liver_model = load('Models/liver_model_2.joblib')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extentsions

# Image preprocessing function
def preprocess_image(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
    img_smooth = cv2.GaussianBlur(img_denoised, (5, 5), 0)
    return img_smooth

# Function to write OCR output to a CSV file
def write_ocr_output_to_csv(ocr_output, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Text', 'Confidence'])  # Write header row
        for text, confidence in ocr_output:
            writer.writerow([text, confidence])

# Perform OCR on image
def perform_ocr_on_image(image):
    img_preprocessed = preprocess_image(image)
    result = ocr.ocr(img_preprocessed, cls=True)
    ocr_results = []
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            ocr_results.append(line[1])  # Append text only
    return ocr_results

# Perform OCR on PDF
def perform_ocr_on_pdf(pdf_bytes):
    imgs = []
    ocr_results = []
    with fitz.open("pdf", pdf_bytes) as pdf:
        for pg in range(pdf.page_count):
            page = pdf[pg]
            pm = page.get_pixmap()
            img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            img_preprocessed = preprocess_image(img)
            result = ocr.ocr(img_preprocessed, cls=True)
            for idx in range(len(result)):
                res = result[idx]
                for line in res:
                    ocr_results.append(line[1])  # Append text only
            imgs.append(img_preprocessed)
    return ocr_results

# Extract glucose value from OCR output CSV
def extract_glucose_value(csv_file):
    glucose_value = None
    with open(csv_file, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] == 'GLUCOSE, RANDOM,':
                glucose_value = next(reader)[0]
                break
    print(glucose_value)
    return glucose_value or None

# Extract lab values from OCR output CSV
def extract_lab_values(csv_file):
    # 2nd try
    lab_values = {}
    with open(csv_file, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        age_gender_index = None
        for row in reader:
            if 'Age/Gender' in row:
                age_gender_index = row.index('Age/Gender')
                break

        if age_gender_index is None:
            raise ValueError("Age/Gender column not found in CSV file.")

        for row in reader:
            age_gender_value = row[age_gender_index].split('/')
            age = int(age_gender_value[0])
            gender = age_gender_value[1]

            lab_values['Age'] = age
            gender = 1 if request.form.get('gender') == 'Male' else 0
            lab_values['Gender'] = gender
            
            break  # Assuming there is only one Age/Gender entry, so breaking after the first occurrence

        for row in reader:
            if row[0] in ['SGOT', 'SGPT', 'ALKALINEPHOSPHATASE', 'BILIRUBIN TOTAL', 'BILIRUBIN DIRECT', 'TOTAL PROTEINS', 'ALBUMIN', 'A:GRATIO']:
                # lab_values[row[0]] = float(row[1])
                lab_values[row[0]] = float(next(reader)[0])
    print(lab_values)
    return lab_values

# Predict diabetes using input data
def predict_diabetes(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = diabetes_model.predict(input_array)
    return prediction[0]

# Predict liver disease using input data
def predict_liver(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = liver_model.predict(input_array)
    return prediction[0]

# Route for processing uploaded file and predicting diabetes
@app.route('/predict', methods=['POST'])
def predict_disease():
    data = request.get_json()
    disease_value = data.get('disease_value')
    image_url = data.get('image_url')

    if not image_url:
        return jsonify({"Error": "Image URL is missing."}), 400

    # Fetch the image from the URL
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image_data = BytesIO(response.content)
    except Exception as e:
        return jsonify({"Error": f"Failed to fetch image from URL: {str(e)}"}), 400

    filename = 'image.png'  # You may want to generate a unique filename here

    if disease_value == 1:
        ocr_output = perform_ocr_on_pdf(image_data) if filename.endswith('.pdf') else perform_ocr_on_image(np.array(Image.open(image_data)))
        output_file = 'ocr_output.csv'
        write_ocr_output_to_csv(ocr_output, output_file)
        lab_values = extract_lab_values(output_file)
        prediction = predict_liver([*lab_values.values()])
        if prediction == 1:
            result = "Congratulations, your liver is HEALTHY."
        else:
            result = "Please visit a doctor. Seems you are unwell."
    
    elif disease_value == 2:
        ocr_output = perform_ocr_on_pdf(image_data) if filename.endswith('.pdf') else perform_ocr_on_image(np.array(Image.open(image_data)))
        output_file = 'ocr_output.csv'
        write_ocr_output_to_csv(ocr_output, output_file)
        glucose_value = extract_glucose_value(output_file)
        prediction = predict_diabetes([0.627, 33.6, 6, 50, glucose_value])
        if prediction == 0:
            result = "Congratulations, you are not DIABETIC."
        else:
            result = "Please visit a doctor. Seems like you are DIABETIC"
    
    else:
        return jsonify({"Error": "Invalid disease value. Valid values are 1 (liver disease) and 2 (diabetes)."})

    return jsonify({"Prediction": result})

if __name__ == "__main__":
    app.run(debug=True)
