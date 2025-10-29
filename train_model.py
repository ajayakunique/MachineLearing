import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# ---------------------------
# 1️⃣ Load dataset
# ---------------------------
data_df = pd.read_csv('heart.csv')

# ---------------------------
# 2️⃣ Data Cleaning
# ---------------------------
# Remove duplicates
data_df.drop_duplicates(inplace=True)

# Handle missing values
# Fill numeric NaNs with mean and categorical NaNs with mode
num_cols = data_df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = data_df.select_dtypes(include=['object']).columns

data_df[num_cols] = data_df[num_cols].fillna(data_df[num_cols].mean())
for col in cat_cols:
    data_df[col] = data_df[col].fillna(data_df[col].mode()[0])

# ---------------------------
# 3️⃣ Encode categorical columns using LabelEncoder
# ---------------------------
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    data_df[col] = le.fit_transform(data_df[col])
    label_encoders[col] = le

# ---------------------------
# 4️⃣ Split features and target
# ---------------------------
x = data_df.drop('target', axis=1)
y = data_df['target']

# ---------------------------
# 5️⃣ Feature scaling
# ---------------------------
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# ---------------------------
# 6️⃣ Train-test split
# ---------------------------
x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, test_size=0.2, random_state=30
)

# ---------------------------
# 7️⃣ Train Logistic Regression model
# ---------------------------
LR = LogisticRegression(solver='liblinear')
LR.fit(x_train, y_train)

# ---------------------------
# 8️⃣ Evaluate model
# ---------------------------
y_pred = LR.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# ---------------------------
# 9️⃣ Save model components
# ---------------------------
pickle.dump(LR, open("heart_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(label_encoders, open("label_encoders.pkl", "wb"))
print("✅ Model, scaler, and label encoders saved successfully!")

# ---------------------------
# 🔟 Predict on test_data.csv
# ---------------------------
test_df = pd.read_csv('test_data.csv')

# Data cleaning for test data
test_df.drop_duplicates(inplace=True)
test_df[num_cols] = test_df[num_cols].fillna(data_df[num_cols].mean())
for col in cat_cols:
    test_df[col] = test_df[col].fillna(data_df[col].mode()[0])

# Apply same label encoding
for col in cat_cols:
    if col in label_encoders:
        le = label_encoders[col]
        test_df[col] = le.transform(test_df[col])

# Drop target if present
if 'target' in test_df.columns:
    test_df = test_df.drop('target', axis=1)

# Apply scaling
test_scaled = scaler.transform(test_df)

# Predict
test_pred = LR.predict(test_scaled)
print("✅ Predictions on test_data.csv:", test_pred)
