
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix

# Title
st.title("Student Mental Health Distress Prediction App")
st.markdown("This Streamlit app predicts mental distress in students based on survey data using a machine learning model.")

# Load dataset
df = pd.read_csv("mentalhealth_dataset.csv")

# Preprocessing
categorical_cols = df.select_dtypes(include='object').columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Feature Engineering
df['TotalDistress'] = df[['Depression', 'Anxiety', 'PanicAttack']].sum(axis=1)
df['Is_Seeking_Help'] = df['SpecialistTreatment']
df['MentalWellnessIndex'] = (df['AcademicEngagement'] + df['HasMentalHealthSupport'] + (5 - df['StudyStressLevel'])) / 3
df['StressSupportRatio'] = df['StudyStressLevel'] / (df['HasMentalHealthSupport'] + 1)
df['Target'] = df['TotalDistress'].apply(lambda x: 1 if x > 0 else 0)

# Standardization
scaler = StandardScaler()
df['Age'] = scaler.fit_transform(df[['Age']])

# PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df.select_dtypes(include='number'))
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

# Features and target
features = ['PCA1', 'PCA2', 'Age', 'Is_Seeking_Help', 'MentalWellnessIndex']
X = df[features]
y = df['Target']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
st.subheader("Model Evaluation")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
st.pyplot(fig)

# Feature Importance
st.subheader("Feature Importance")
importance = pd.Series(model.feature_importances_, index=X.columns)
fig2, ax2 = plt.subplots()
importance.sort_values().plot(kind='barh', ax=ax2)
plt.title("Feature Importance")
st.pyplot(fig2)

# Prediction Section
st.subheader("Make a Prediction")
with st.form("prediction_form"):
    age = st.slider("Age (standardized)", float(df['Age'].min()), float(df['Age'].max()), 0.0)
    seeking_help = st.selectbox("Is Seeking Help?", [0, 1])
    wellness_index = st.slider("Mental Wellness Index", float(df['MentalWellnessIndex'].min()), float(df['MentalWellnessIndex'].max()), 1.0)
    pca1 = st.slider("PCA1", float(df['PCA1'].min()), float(df['PCA1'].max()), 0.0)
    pca2 = st.slider("PCA2", float(df['PCA2'].min()), float(df['PCA2'].max()), 0.0)
    submit = st.form_submit_button("Predict")

if submit:
    input_data = pd.DataFrame([[pca1, pca2, age, seeking_help, wellness_index]], columns=features)
    prediction = model.predict(input_data)[0]
    result = "Distressed" if prediction == 1 else "Not Distressed"
    st.success(f"The student is predicted to be: **{result}**")
