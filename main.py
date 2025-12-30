import pandas as pd
import joblib
import yaml
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. טעינת הנתונים
df = pd.read_csv('parkinsons.csv')

# 2. בחירת המאפיינים (Features) ועמודת המטרה (Target)
# נסיר עמודות לא רלוונטיות לחישוב
X = df.drop(columns=['status', 'name'])
y = df['status']

# 3. נרמול הנתונים (Scaling)
# מומלץ לנרמל את כל ה-X כדי להבטיח דיוק גבוה מ-0.8
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

# 4. חלוקה למערך אימון ותיקוף
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. בחירת מודל ואימון (SVM כפי שהוצע במאמר)
model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train, y_train)

# 6. הערכת המודל
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Final Model Accuracy: {accuracy:.4f}")

# 7. שמירת המודל ועדכון קובץ ה-Config
model_filename = 'parkinsons_model.joblib'
joblib.dump(model, model_filename)

selected_features = X.columns.tolist()
config_data = {
    'model_name': model_filename,
    'features': selected_features
}

with open('config.yaml', 'w') as file:
    yaml.dump(config_data, file)

print("Files main.py, parkinsons_model.joblib, and config.yaml are ready.")
