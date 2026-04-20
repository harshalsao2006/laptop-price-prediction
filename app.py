import streamlit as st
import pandas as pd
import pickle

# Page config
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="centered")

st.title("💻 Laptop Price Prediction")
st.write("Enter laptop specifications to predict the price.")

# Model load
with open("pipe.pkl", "rb") as f:
    pipe = pickle.load(f)

# Dropdown options
companies = ["Dell", "HP", "Lenovo", "Asus", "Acer", "Apple", "MSI"]
types = ["Notebook", "Ultrabook", "Gaming", "2 in 1 Convertible", "Workstation"]
cpu_brands = ["Intel Core i3", "Intel Core i5", "Intel Core i7", "AMD Ryzen 5", "AMD Ryzen 7"]
gpu_brands = ["Intel", "Nvidia", "AMD"]
oses = ["Windows", "Mac", "Linux", "DOS"]

# Inputs
company = st.selectbox("Brand", companies)
typename = st.selectbox("Laptop Type", types)
ram = st.selectbox("RAM (GB)", [4, 8, 16, 32, 64])
weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, step=0.1)
touchscreen = st.selectbox("Touchscreen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
ips = st.selectbox("IPS Display", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
ppi = st.number_input("PPI", min_value=50, max_value=400, step=1)
cpu_brand = st.selectbox("CPU Brand", cpu_brands)
hdd = st.selectbox("HDD (GB)", [0, 128, 256, 500, 1000, 2000])
ssd = st.selectbox("SSD (GB)", [0, 128, 256, 512, 1024, 2048])
gpu_brand = st.selectbox("GPU Brand", gpu_brands)
os = st.selectbox("Operating System", oses)

if st.button("Predict Price"):
    input_df = pd.DataFrame([{
        "Company": company,
        "TypeName": typename,
        "Ram": ram,
        "Weight": weight,
        "Touchscreen": touchscreen,
        "Ips": ips,
        "PPI": ppi,
        "Cpu_brand": cpu_brand,
        "HDD": hdd,
        "SSD": ssd,
        "Gpu_brand": gpu_brand,
        "Os": os
    }])

    prediction = pipe.predict(input_df)[0]
    st.success(f"Estimated Laptop Price: ₹ {round(prediction, 2)}")