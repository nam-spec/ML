# ==========================================
# Step 2: Import Libraries
# ==========================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from xgboost import XGBClassifier

# ==========================================
# Step 3: Upload Dataset
# ==========================================

from google.colab import files
uploaded = files.upload()


# ==========================================
# Step 4: Load Dataset
# ==========================================

df = pd.read_csv('Titanic-Dataset.csv')

print(df.head())

# ==========================================
# Step 5: Data Preprocessing
# ==========================================

# Drop unnecessary columns
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].mean())

df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Encode categorical columns
le = LabelEncoder()

df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

# ==========================================
# Step 6: Features and Target
# ==========================================

X = df.drop('Survived', axis=1)

y = df['Survived']

# ==========================================
# Step 7: Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# Step 8: Create XGBoost Classifier
# ==========================================

model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# ==========================================
# Step 9: Predictions
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# Step 10: Evaluation
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))