import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd

app = Flask(__name__)

# Define the model
class Anomaly_Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Anomaly_Classifier, self).__init__()
        self.conv = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=5, stride=1)
        self.conv_pad = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.maxpool = nn.MaxPool1d(kernel_size=5, stride=2)
        
        # Placeholder for dynamically calculated flattened size
        self.flattened_size = None
        
        # Fully connected layers
        self.dense1 = nn.Linear(32 * 8, 32)  # This will be updated later
        self.dense2 = nn.Linear(32, 32)
        self.dense_final = nn.Linear(32, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        residual = self.conv(x)
        x = F.relu(self.conv_pad(residual))
        x = self.conv_pad(x)
        x += residual
        x = F.relu(x)
        residual = self.maxpool(x)
        
        x = F.relu(self.conv_pad(residual))
        x = self.conv_pad(x)
        x += residual
        x = F.relu(x)
        residual = self.maxpool(x)
        
        x = F.relu(self.conv_pad(residual))
        x = self.conv_pad(x)
        x += residual
        x = F.relu(x)
        residual = self.maxpool(x)
        
        x = F.relu(self.conv_pad(residual))
        x = self.conv_pad(x)
        x += residual
        x = F.relu(x)
        x = self.maxpool(x)
        
        # Calculate the flattened size dynamically if it hasn't been set
        if self.flattened_size is None:
            self.flattened_size = x.view(x.size(0), -1).size(1)
            # Update the first fully connected layer based on the calculated flattened size
            self.dense1 = nn.Linear(self.flattened_size, 32)
        
        x = x.view(x.size(0), -1)  # Flatten for the fully connected layers
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.softmax(self.dense_final(x))
        return x

# Load the model
anom_classifier = Anomaly_Classifier(input_size=1, num_classes=5)
anom_classifier.load_state_dict(torch.load('Dev Process\\model\\anom_classifier.pth', map_location=torch.device('cpu')))
anom_classifier.eval()

# Preprocess ECG data (pad/trim to the required length)
def preprocess_ecg_signal(ecg_signal, target_length=256):
    # Trim or pad to match the target length
    if len(ecg_signal) > target_length:
        ecg_signal = ecg_signal[:target_length]  # Trim to required length
    elif len(ecg_signal) < target_length:
        padding = target_length - len(ecg_signal)
        ecg_signal = np.pad(ecg_signal, (0, padding), 'constant')
    
    # Normalize the signal
    ecg_signal = ecg_signal / np.max(np.abs(ecg_signal))
    
    # Reshape to match model input: (batch_size, channels, sequence_length)
    ecg_tensor = torch.tensor(ecg_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, target_length)
    
    return ecg_tensor

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Read CSV file
        df = pd.read_csv(file)
        results = []

        for index, row in df.iterrows():
            ecg_signal = np.array(row)  # Convert row to NumPy array

            # Preprocess the signal to match model input
            ecg_tensor = preprocess_ecg_signal(ecg_signal, target_length=256)

            # Make predictions using the loaded model
            with torch.no_grad():
                prediction = anom_classifier(ecg_tensor)

            class_labels = {
                0: 'Normal Heartbeat',
                1: 'Supra Ventricular Premature Beat',
                2: 'Premature Ventricular Beat',
                3: 'Fusion Beat',
                4: 'Unknown Beat'
            }

            # Get the predicted class
            predicted_class = prediction.argmax().item()
            prediction_label = class_labels.get(predicted_class, "Unknown")
            results.append({
                'index': index,
                'prediction': prediction_label
            })

        # Log the results for debugging
        print("Predictions:", results)

        # Pass the predictions to the template for display
        return render_template('index.html', result=results)

    except Exception as e:
        # Log the error message for debugging
        print("Error:", str(e))
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
