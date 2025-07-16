import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("iris_model.pkl")

# Title
st.title(" Iris Flower Prediction App")

# Sidebar - User Input
st.sidebar.header("Input Features")
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.2)

# Prepare input data
input_df = pd.DataFrame({
    'sepal length (cm)': [sepal_length],
    'sepal width (cm)': [sepal_width],
    'petal length (cm)': [petal_length],
    'petal width (cm)': [petal_width]
})

# Prediction
prediction = model.predict(input_df)[0]
classes = ['Setosa', 'Versicolor', 'Virginica']

# Output
st.subheader("Prediction:")
st.write(f" The predicted flower is **{classes[prediction]}**")

# Show input data
st.subheader("Input Summary:")
st.write(input_df)

# Feature Importance Plot
st.subheader("Feature Importance:")
importances = model.feature_importances_
features = input_df.columns

plt.barh(features, importances)
plt.xlabel("Importance")
plt.ylabel("Feature")
st.pyplot(plt)