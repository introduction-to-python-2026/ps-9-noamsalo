import pandas as pd
import joblib
import yaml
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

# 1. טעינת הנתונים
df = pd.read_csv('parkinsons.csv')

# 2. בחירת בדיוק 2 משתנים (לפי דרישת הטסט: assert len(features) == 2)
# נשתמש בשניים נפוצים:
selected_features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)']
X = df[selected_features]
y = df['status']

# 3. נרמול (Scaling) - קריטי כדי להגיע לדיוק הנדרש
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
# חשוב להחזיר ל-DataFrame כדי שהמודל "יכיר" את שמות העמודות בטסט
X_scaled_df = pd.DataFrame(X_scaled, columns=selected_features)

# 4. אימון מודל (SVM עם פרמטרים חזקים)
model = SVC(kernel='rbf', C=10.0, gamma='scale')
model.fit(X_scaled_df, y)

# 5. שמירת המודל
model_filename = 'parkinsons_model.joblib'
joblib.dump(model, model_filename)

# 6. יצירת קובץ config.yaml בפורמט שהטסט דורש
config_data = {
    'path': model_filename,    # הטסט מחפש 'path'
    'features': selected_features
}

with open('config.yaml', 'w') as file:
    yaml.dump(config_data, file, default_flow_style=False)

print("Main.py execution finished. Model and Config are ready for testing.")
