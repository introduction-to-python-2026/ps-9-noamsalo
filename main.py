import pandas as pd
import joblib
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# 1. Load the dataset
df = pd.read_csv('parkinsons.csv')

# 2. Select features (העמודות שבחרת ב-Notebook)
selected_features = ['PPE', 'MDVP:Fo(Hz)']
X = df[selected_features]
y = df['status']

# 4. Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 3 + 5. Scale and Model (שימוש ב-Pipeline - זה הפתרון!)
# ה-Pipeline דואג שה-MinMaxScaler יופעל אוטומטית בתוך המודל
model = Pipeline([
    ('scaler', MinMaxScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=3))
])

# אימון ה-Pipeline
model.fit(X_train, y_train)

# 6. Test accuracy
acc = accuracy_score(y_val, model.predict(X_val))
print(f"Accuracy: {acc}")

# 7. Save the model
model_path = 'my_model.joblib'
joblib.dump(model, model_path)

# 7. Update config.yaml (בפורמט המדויק שהדוגמה ביקשה)
config_data = {
    'selected_features': selected_features,
    'path': model_path
}

with open('config.yaml', 'w') as file:
    yaml.dump(config_data, file, default_flow_style=False)

print("Done! Upload main.py, my_model.joblib, and config.yaml to GitHub.")
