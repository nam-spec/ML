# Non-Linear SVM using RBF Kernel

# Install scikit-learn
# !pip install scikit-learn

# ==========================================
# NON-LINEAR SVM (RBF KERNEL)
# TITANIC DATASET
# ==========================================

# Step 1: Import Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# Step 2: Upload Dataset
# ==========================================

from google.colab import files
uploaded = files.upload()

# ==========================================
# Step 3: Load Dataset
# ==========================================

df = pd.read_csv('Titanic-Dataset.csv')

print(df.head())


# ==========================================
# Step 4: Data Preprocessing
# ==========================================

# Drop unnecessary columns
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Fill missing values
df['Age'] = df['Age'].fillna(df['Age'].mean())

df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Encoding categorical columns
le = LabelEncoder()

df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

# ==========================================
# Step 5: Features and Target
# ==========================================

X = df.drop('Survived', axis=1)
y = df['Survived']

# ==========================================
# Step 6: Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# Step 7: Feature Scaling
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================
# Step 8: Create Non-Linear SVM Model
# ==========================================

model = SVC(
    kernel='rbf',      # Non-linear kernel
    C=1,
    gamma='scale',
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