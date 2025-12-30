import pandas as pd
import joblib
import yaml
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. טעינת הנתונים - ודא שהקובץ נמצא בתיקייה הראשית ב-GitHub
file_path = 'parkinsons.csv'
if not os.path.exists(file_path):
    # ניסיון טעינה למקרה שהקובץ בנתיב אחר
    file_path = 'parkinsons.csv'

df = pd.read_csv(file_path)

# 2. ניקוי נתונים בסיסי - הסרת עמודות לא נומריות
# בדרך כלל יש עמודת 'name' שצריך להסיר
X = df.drop(columns=['status'])
if 'name' in X.columns:
    X = X.drop(columns=['name'])

y = df['status']

# 3. נרמול - קריטי להגיע לדיוק מעל 0.8
scaler = MinMaxScaler()
# חשוב: אנחנו מנרמלים את כל העמודות כדי שהמודל יצליח ללמוד
X_scaled = scaler.fit_transform(X)
X_final = pd.DataFrame(X_scaled, columns=X.columns)

# 4. חלוקה למערך אימון ותיקוף
X_train, X_val, y_train, y_val = train_test_split(X_final, y, test_size=0.2, random_state=42)

# 5. בחירת מודל - SVM עם פרמטרים שמשפרים דיוק
model = SVC(kernel='linear', C=1.0) # Kernel ליניארי עובד מצוין על הסט הזה
model.fit(X_train, y_train)

# 6. בדיקת דיוק
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy: {accuracy}")

# 7. שמירת הקבצים בדיוק לפי הדרישות
model_filename = 'parkinsons_model.joblib'
joblib.dump(model, model_filename)

# יצירת רשימת המאפיינים עבור ה-config
config_data = {
    'model_name': model_filename,
    'features': X.columns.tolist()
}

with open('config.yaml', 'w') as file:
    yaml.dump(config_data, file, default_flow_style=False)

print("Process finished successfully!")
