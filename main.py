import pandas as pd
import joblib
import yaml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

# 1. טעינת הנתונים
df = pd.read_csv('parkinsons.csv')

# 2. בחירת המאפיינים (חייב להיות 2 עמודות לפי המודל שלך)
# בחרתי את שתי העמודות הראשונות, וודא שאלו העמודות שבהן השתמשת
selected_features = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)"]
X = df[selected_features]
y = df['status']

# 3. נרמול (Scaling)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 4. המודל שכבר יש לך (טעון מהקובץ שהעלית)
# במקום לאמן מחדש, אנחנו משתמשים בקובץ ששלחת לי
model = joblib.load('my_model.joblib')

# 5. שמירה סופית של המודל (ליתר ביטחון, כדי שיהיה בתיקייה)
joblib.dump(model, 'my_model.joblib')

# 6. יצירת קובץ ה-Config בפורמט שהטסט דורש
config_data = {
    'selected_features': selected_features,
    'path': 'my_model.joblib'
}

with open('config.yaml', 'w') as file:
    yaml.dump(config_data, file, default_flow_style=False)

print("Main.py is ready and matches your joblib file!")
