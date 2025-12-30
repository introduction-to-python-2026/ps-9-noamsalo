import pandas as pd
import joblib
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# 1. Load the dataset
df = pd.read_csv('parkinsons.csv')

# 2. Select features (2 inputs, 1 output)
# Based on the paper, Fo and Fhi are strong indicators
selected_features = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)"]
target_feature = "status"

X = df[selected_features]
y = df[target_feature]

# 4. Split the data (Training and Validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 3 & 5. Scale the data & Choose a model
# We use a Pipeline to bundle MinMaxScaler and SVC together.
# This ensures that when the test script calls model.predict(), 
# the scaling is applied automatically to the raw test data.
model = Pipeline([
    ('scaler', MinMaxScaler()),
    ('svm', SVC(kernel='linear', C=10.0))
])

# Training
model.fit(X_train, y_train)

# 6. Test the accuracy
y_pred = model.predict(X_val)
score = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {score:.4f}")

# 7. Save the model and update config.yaml
model_filename = 'parkinsons_model.joblib'
joblib.dump(model, model_filename)

# Note: The test script expects 'path' and 'features' keys
config_data = {
    'selected_features': selected_features,
    'path': model_filename
}

with open('config.yaml', 'w') as file:
    yaml.dump(config_data, file, default_flow_style=False)

print("Setup complete. main.py, model, and config are ready.")
