import pandas as pd
import joblib
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

# 1. טעינת הנתונים
df = pd.read_csv('parkinsons.csv')

# 2. בחירת 2 משתנים (חובה לפי הטסט)
selected_features = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)"]
X = df[selected_features]
y = df['status']

# 3. יצירת Pipeline (זה הסוד שיפתור את האיקס!)
# ה-Pipeline דואג שהנרמול יקרה אוטומטית גם בזמן הטסט
model = Pipeline([
    ('scaler', MinMaxScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])

# 4. אימון המודל
model.fit(X, y)

# 5. שמירת המודל בשם המדויק שרצית
model_filename = 'parkinsons_model-2.joblib'
joblib.dump(model, model_filename)

# 6. יצירת קובץ ה-config בדיוק בפורמט שהטסט מחפש
config_data = {
    'path': model_filename,
    'features': selected_features
}

with open('config.yaml', 'w') as file:
    yaml.dump(config_data, file, default_flow_style=False)

print("Final version saved successfully!")
