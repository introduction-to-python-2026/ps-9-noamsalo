import pandas as pd
import joblib
import yaml
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

# בדיקה איזה קובץ קיים בתיקייה (למנוע שגיאת FileNotFoundError)
if os.path.exists('parkinsons.csv'):
    file_name = 'parkinsons.csv'
elif os.path.exists('parkinson.csv'):
    file_name = 'parkinson.csv'
else:
    file_name = 'parkinsons.csv' # ברירת מחדל

# 1. טעינה
df = pd.read_csv(file_name)

# 2. בחירת עמודות (בדיוק 2 לפי הטסט)
# אם השמות האלו לא קיימים בדיוק ככה, הטסט יכשל
features = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)"]
X = df[features]
y = df['status']

# 3. יצירת Pipeline (חובה כדי שהנרמול ישמר במודל)
model = Pipeline([
    ('scaler', MinMaxScaler()),
    ('svm', SVC(kernel='linear', C=10.0))
])

# 4. אימון
model.fit(X, y)

# 5. שמירה (בשם שהדוגמה ביקשה)
model_path = 'parkinsons_model.joblib'
joblib.dump(model, model_path)

# 6. יצירת הקונפיג בדיוק לפי הפורמט של הדוגמה
config_data = {
    'selected_features': features,
    'path': model_path
}

with open('config.yaml', 'w') as f:
    yaml.dump(config_data, f, default_flow_style=False)

print("Done! Everything is ready.")
