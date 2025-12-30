 import pandas as pd
import joblib
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# שלב 1: טעינת הנתונים
df = pd.read_csv('parkinsons.csv')

# שלב 2: בחירת משתנים (2 קלט, 1 פלט)
# בחרנו שני מאפיינים חזקים לפי הספרות המקצועית
selected_features = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)"]
X = df[selected_features]
y = df['status']

# שלב 4: חלוקה למערך אימון ותיקוף
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# שלב 3 + 5: יצירת מודל הכולל נרמול (Pipeline)
# ה-Pipeline מבטיח שה-MinMaxScaler יופעל גם על נתוני הטסט הסודיים
model = Pipeline([
    ('scaler', MinMaxScaler()),
    ('svm', SVC(kernel='linear', C=10.0))
])

# אימון המודל
model.fit(X_train, y_train)
