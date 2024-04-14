import tkinter as tk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
url = r"C:\Users\sanna\Downloads\parkinsons.csv"
df = pd.read_csv(url)

# Select features and target
features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 
            'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 
            'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 
            'D2', 'PPE']
X = df[features]
y = df['status']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)

# Create a Tkinter GUI
root = tk.Tk()
root.title("Parkinson's Disease Prediction")

# Create a label and entry field for pasting all features at once
tk.Label(root, text="Enter all features separated by spaces:").grid(row=0, column=0, sticky=tk.W)
all_features_entry = tk.Entry(root, width=50)
all_features_entry.grid(row=0, column=1)

# Create a button to set individual feature values
def set_input_features():
    all_features = all_features_entry.get().split()
    for i, val in enumerate(all_features):
        feature_entries[i].delete(0, tk.END)
        feature_entries[i].insert(0, val)

set_features_button = tk.Button(root, text="Set Features", command=set_input_features)
set_features_button.grid(row=0, column=2)

# Create labels and entry fields for feature values
feature_entries = []
for i, feature in enumerate(features):
    tk.Label(root, text=feature).grid(row=i+1, column=0, sticky=tk.W)
    entry = tk.Entry(root)
    entry.grid(row=i+1, column=1)
    feature_entries.append(entry)

# Function to get feature values from the entry fields
def get_feature_values():
    return [float(entry.get()) for entry in feature_entries]

# Function to predict Parkinson's disease
def predict_parkinsons():
    feature_values = get_feature_values()
    prediction = rf_classifier.predict([feature_values])
    result_label.config(text="The prediction indicates Parkinson's disease." if prediction[0] == 1 else "The prediction indicates no Parkinson's disease.")

# Button to trigger prediction
predict_button = tk.Button(root, text="Predict", command=predict_parkinsons)
predict_button.grid(row=len(features)+1, column=0, columnspan=2)

# Label to display prediction result
result_label = tk.Label(root, text="")
result_label.grid(row=len(features)+2, column=0, columnspan=2)

root.mainloop()
