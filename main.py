import pandas as pd
import joblib
import yaml
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

# 1. טעינת הנתונים
df = pd.read_csv('parkinsons.csv')

# 2. בחירת בדיוק 2 משתנים (חובה לפי דרישת הטסט)
selected_features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)']
X = df[selected_features]
y = df['status']

# 3. יצירת Pipeline שכולל גם נרמול וגם מודל
# זה מבטיח שהטסט יצליח להריץ את המודל על נתונים גולמיים
model_pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('svm', SVC(kernel='linear', C=10.0))
])

# 4. אימון ה-Pipeline
model_pipeline.fit(X, y)

# 5. שמירת ה-Pipeline כקובץ המודל
model_filename = 'parkinsons_model.joblib'
joblib.dump(model_pipeline, model_filename)

# 6. יצירת קובץ config.yaml בפורמט שהטסט דורש
config_data = {
    'path': model_filename,
    'features': selected_features
}

with open('config.yaml', 'w') as file:
    yaml.dump(config_data, file, default_flow_style=False)

print("Pipeline model and config are ready!")
