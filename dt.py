# Step 1: Import Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from google.colab import files

uploaded = files.upload()

df = pd.read_csv('Titanic-Dataset.csv')

print(df.head())


# ==========================================
# Step 3: Data Preprocessing
# ==========================================

# Drop unnecessary columns
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Fill missing Age values with mean
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Fill missing Embarked values with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Convert categorical columns into numerical values
le = LabelEncoder()

df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

# ==========================================
# Step 4: Define Features and Target
# ==========================================

X = df.drop('Survived', axis=1)
y = df['Survived']

# ==========================================
# Step 5: Split Dataset
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# Step 6: Create Decision Tree Model
# ==========================================

model = DecisionTreeClassifier(
    criterion='gini',     # or 'entropy'
    max_depth=5,
    random_state=42
)


# Train model
model.fit(X_train, y_train)

# ==========================================
# Step 7: Predictions
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# Step 8: Evaluation
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

