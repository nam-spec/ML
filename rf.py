
#rf classification


# Step 1: Import Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
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

# Encode categorical columns
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
# Step 7: Create Random Forest Model
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,     # number of trees
    criterion='gini',     # or 'entropy'
    max_depth=5,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# ==========================================
# Step 8: Predictions
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# Step 9: Evaluation
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))



#rf regression


# ==========================================
# RANDOM FOREST REGRESSION
# CALIFORNIA HOUSING DATASET
# Google Colab Ready Code
# ==========================================

# Step 1: Import Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==========================================
# Step 2: Upload Dataset
# ==========================================

from google.colab import files
uploaded = files.upload()

# ==========================================
# Step 3: Load Dataset
# ==========================================

df = pd.read_csv('housing.csv')

print(df.head())


# ==========================================
# Step 4: Handle Missing Values
# ==========================================

df['total_bedrooms'] = df['total_bedrooms'].fillna(
    df['total_bedrooms'].mean()
)

# ==========================================
# Step 5: Convert Categorical Data
# ==========================================

df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)

# ==========================================
# Step 6: Features and Target
# ==========================================

X = df.drop('median_house_value', axis=1)

y = df['median_house_value']

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
# Step 8: Create Random Forest Regressor
# ==========================================

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

# Train Model
model.fit(X_train, y_train)

# ==========================================
# Step 9: Predictions
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# Step 10: Evaluation
# ==========================================

mse = mean_squared_error(y_test, y_pred)

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error:", mse)

print("\nMean Absolute Error:", mae)

print("\nRoot Mean Squared Error:", rmse)

print("\nR2 Score:", r2)

