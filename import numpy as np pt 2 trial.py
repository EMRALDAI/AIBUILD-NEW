import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

# Step 3: Data Collection and Preprocessing
# Generate synthetic data for demonstration
np.random.seed(42)
normal_data = np.random.normal(loc=0, scale=1, size=(1000, 3))
anomalous_data = np.random.normal(loc=5, scale=1, size=(50, 3))
data = np.vstack([normal_data, anomalous_data])
labels = np.hstack([np.zeros(1000), np.ones(50)])

# Create a DataFrame
df = pd.DataFrame(data, columns=['feature1', 'feature2', 'feature3'])
df['label'] = labels

# Save the dataset to a CSV file
df.to_csv('synthetic_dataset.csv', index=False)

# Load your dataset
data = pd.read_csv('synthetic_dataset.csv')

# Preprocess the data
data.fillna(0, inplace=True)  # Handling missing values

# Step 4: Feature Engineering
# Extract features and labels
features = data[['feature1', 'feature2', 'feature3']]
labels = data['label']

# Step 5: Model Selection and Training
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize and train the Isolation Forest model
model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
model.fit(X_train)

# Step 6: Model Evaluation
# Predict on the test set
y_pred = model.predict(X_test)
y_pred = [1 if x == -1 else 0 for x in y_pred]  # Convert -1 to 1 for anomalies

# Evaluate the model
print(classification_report(y_test, y_pred))

# Step 7: Real-Time Detection
def detect_anomaly(new_data):
    prediction = model.predict(new_data)
    if prediction == -1:
        print("Anomaly detected!")
    else:
        print("No anomaly detected.")

# Example usage with new data
new_data = pd.DataFrame([[5, 5, 5]], columns=['feature1', 'feature2', 'feature3'])
detect_anomaly(new_data)
