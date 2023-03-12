import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('breast_cancer_data.csv')

# Split the dataset into features and target variable
X = df.drop(['diagnosis'], axis=1)
y = df['diagnosis']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Build the model
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
clf.fit(X_train, y_train)
def predict_cancer(data):
    """
    Predict whether a patient has cancer or not based on their data
    """
    # Convert the input data to a numpy array
    data = np.array([data])

    # Make a prediction
    prediction = clf.predict(data)

    # Return the prediction
    return prediction[0]
