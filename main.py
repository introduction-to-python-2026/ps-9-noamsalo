import pandas as pd
import joblib
import yaml
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

# 1. טעינת הנתונים (בדיקה איזה שם קובץ קיים ב-GitHub שלך)
if os.path.exists('parkinsons.csv'):
    file_name = 'parkinsons.csv'
else:
    file_name = 'parkinson.csv'

df = pd.read_csv(file_name)

# 2. בחירת 2 עמודות (חובה לפי הטסט)
features = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)"]
X = df[features]
y = df['status']

# 3. יצירת Pipeline (כולל נרמול ומודל)
model = Pipeline([
    ('scaler', MinMaxScaler()),
    ('svm', SVC(kernel='linear', C=10.0))
])

# 4. אימון
model.fit(X, y)

# 5. שמירה בשם החדש שקבעת
model_path = 'parkinsons_model-2.joblib'
joblib.dump(model, model_path)

# 6. עדכון ה-Config עם השם החדש
config_data = {
    'selected_features': features,
    'path': model_path  # כאן אנחנו מעדכנים ל-parkinsons_model-2.joblib
}

with open('config.yaml', 'w') as f:
    yaml.dump(config_data, f, default_flow_style=False)

print(f"Success! Model saved as {model_path} and config updated.")
