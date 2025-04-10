import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# setting the page configurations
st.set_page_config(
    page_title="Home | T.P.M.",
)

# sidebar for hovering through pages

st.sidebar.text('')

# Home Page is divided into 6 sections
# - Header
# - Short Welcome Message
# - What this App does
# - About the dataset
# - Quick Summary Stats
# - Navigation Prompt

# 1. Page Title + Welcome 
st.title('🧠 Thyroid Health Predictor')
st.markdown("""
Welcome to the **Thyroid Health Predictor** — a smart web application that helps you explore thyroid-related health patterns, visualize medical data, and get accurate predictions on thyroid conditions.

Use this app to:
- 🔍 Understand thyroid disorder types
- 📊 Explore thyroid-related health metrics
- 🤖 Get a prediction based on your medical data


""")

st.markdown("<hr style='border:1px solid #ccc;'>", unsafe_allow_html=True)
# 2. Dataset Overview + Key Stats
st.header("📂 About the Dataset")
st.markdown("""
We use a **synthetic dataset of 6,000 patients**, each with:
- 🧪 Hormone levels: TSH, T3, T4
- ❤️ Health markers: Heart rate, cholesterol, glucose, vitamin D
- 🧬 Genetic & lifestyle factors: Family history, smoking, goiter
- 🚻 Demographics: Age, gender, BMI, menstrual cycle

Each record is labeled with one of **five thyroid conditions**.


""")

col1, col2, col3 = st.columns(3)
col1.metric("🧬 Records", "6,000")
col2.metric("📊 Features", "16")
col3.metric("🩺 Thyroid Types", "5")

csv_file = pd.read_csv('./thyroid_dataset.csv')
st.write(csv_file)

st.markdown("<hr style='border:1px solid #ccc;'>", unsafe_allow_html=True)
# 3. Thyroid Type Details
st.header("🔎 Types of Thyroid Conditions")

thyroid_types = {
    "Euthyroid": "Normal thyroid function. Hormone levels are balanced, and there are no signs of disorder.",
    "Hypothyroidism": "Underactive thyroid. TSH is high, T3/T4 are low. Can cause fatigue, weight gain, depression.",
    "Hyperthyroidism": "Overactive thyroid. TSH is low, T3/T4 are high. Leads to anxiety, weight loss, fast heart rate.",
    "Subclinical Hypothyroid": "Slightly elevated TSH, normal T3/T4. Early stage of hypothyroidism with subtle symptoms.",
    "Subclinical Hyperthyroid": "Slightly low TSH, normal T3/T4. Often symptomless but may lead to long-term issues."
}

for condition, desc in thyroid_types.items():
    st.subheader(f"🌀 {condition}")
    st.markdown(desc)
    
st.markdown("<hr style='border:1px solid #ccc;'>", unsafe_allow_html=True)
# 4. Thyroid Type Distribution
st.subheader("📈 Thyroid Type Proportion (Pie Chart)")
type_counts = csv_file["THYROID_TYPE"].value_counts()
fig1, ax1 = plt.subplots()
ax1.pie(type_counts, labels=type_counts.index, autopct="%1.1f%%", startangle=90, colors=plt.cm.Pastel1.colors)
ax1.axis("equal")
st.pyplot(fig1)


st.markdown("<hr style='border:1px solid #ccc;'>", unsafe_allow_html=True)
# 5. Navigation Tips
st.header("🚀 Explore the App")
st.markdown("""
- Go to the **Analysis** page to explore feature distributions and relationships.
- Use the **Report** page to input your details and get a personalized prediction.
""")
