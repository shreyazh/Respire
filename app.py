import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set up Streamlit app
st.set_page_config(page_title="Respire", layout="wide")
st.title("Respire: Solutions for Respiratory Diseases")

# Sidebar for navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to:", ["Home", "FibroPred: Pulmonary Fibrosis Prediction", "QResp: QR Therapeutic Tool", "Data Visualization", "Patient Symptom Tracker"])

if option == "Home":
    st.header("Welcome to the Respiratory Diseases Hackathon App!")
    st.write("This app provides solutions to challenges posed by respiratory diseases.")
    st.markdown("### Challenges Addressed:")
    st.markdown("1. **FibroPred**: Predictive modeling for pulmonary fibrosis.")
    st.markdown("2. **QResp**: QR-based therapeutic decision support.")
    st.markdown("3. **Data Visualization**: Interactive plots for exploring data trends.")
    st.markdown("4. **Patient Symptom Tracker**: Log and analyze patient symptoms over time.")
    st.image("Respire_20241215_130022_0000.png", caption="Innovating for better respiratory health.")

elif option == "FibroPred: Pulmonary Fibrosis Prediction":
    st.header("FibroPred: Pulmonary Fibrosis Prediction")

    # File upload
    uploaded_file = st.file_uploader("Upload a dataset for Pulmonary Fibrosis", type=["csv"])

    if uploaded_file is not None:
        # Load dataset
        data = pd.read_csv(uploaded_file)
        st.write("### Dataset Overview:")
        st.dataframe(data.head())

        # Select features and target
        st.write("### Select Features and Target:")
        features = st.multiselect("Select Features:", options=data.columns)
        target = st.selectbox("Select Target:", options=data.columns)

        if features and target:
            X = data[features]
            y = data[target]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train model
            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            st.write(f"### Model Accuracy: {acc:.2f}")
            st.write(f"### Precision: {precision:.2f}")
            st.write(f"### Recall: {recall:.2f}")
            st.write(f"### F1 Score: {f1:.2f}")

            # Feature importance
            st.write("### Feature Importance:")
            importance = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_})
            st.bar_chart(importance.set_index("Feature"))

elif option == "QResp: QR Therapeutic Tool":
    st.header("QResp: QR Therapeutic Decision Tool")

    # Input patient details
    st.write("### Enter Patient Details:")
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    symptoms = st.text_area("Symptoms (comma-separated)")
    history = st.text_area("Medical History (comma-separated)")

    # Generate QR Code
    if st.button("Generate QR Code"):
        qr_data = f"Age: {age}\nSymptoms: {symptoms}\nHistory: {history}"
        try:
            import qrcode
            from PIL import Image
            qr = qrcode.make(qr_data)
            st.image(qr, caption="Therapeutic QR Code")
        except ImportError:
            st.error("Please install the `qrcode` and `Pillow` libraries to generate QR codes.")

elif option == "Data Visualization":
    st.header("Data Visualization")
    uploaded_file = st.file_uploader("Upload a dataset for visualization", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("### Dataset Overview:")
        st.dataframe(data.head())

        # Correlation heatmap
        st.write("### Correlation Heatmap:")
        corr = data.corr()
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        st.pyplot(plt)

        # Scatter plot
        st.write("### Scatter Plot:")
        x_axis = st.selectbox("Select X-axis:", options=data.columns)
        y_axis = st.selectbox("Select Y-axis:", options=data.columns)
        if x_axis and y_axis:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=data, x=x_axis, y=y_axis)
            st.pyplot(plt)

elif option == "Patient Symptom Tracker":
    st.header("Patient Symptom Tracker")

    # Symptom logging
    st.write("### Log Symptoms:")
    name = st.text_input("Patient Name")
    symptom = st.text_area("Symptoms")
    timestamp = st.date_input("Date")

    if st.button("Save Log"):
        if name and symptom and timestamp:
            with open("symptom_logs.csv", "a") as f:
                f.write(f"{name},{symptom},{timestamp}\n")
            st.success("Log saved successfully.")
        else:
            st.error("Please fill in all fields.")

    # View logs
    st.write("### View Symptom Logs:")
    try:
        logs = pd.read_csv("symptom_logs.csv", names=["Name", "Symptoms", "Date"])
        st.dataframe(logs)
    except FileNotFoundError:
        st.write("No logs found. Start logging symptoms!")

st.sidebar.info("Developed for the Respiratory Diseases Hackathon.")
