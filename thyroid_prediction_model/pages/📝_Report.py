import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import streamlit.components.v1 as components


# setting the page configurations
st.set_page_config(
    page_title="Report | T.P.M.",
    # page_icon="./tpm_icon.png",
    layout="wide"
)
# st.sidebar.image("company_icon.png", use_column_width=True)

st.title("📝 Thyroid Health Report")
st.markdown("This report includes ETL, EDA, predictions, and model evaluation based on your input.")

# LOAD DATA 
st.subheader("📥 Data Extraction")
df = pd.read_csv("thyroid_dataset.csv")
st.dataframe(df.head())

st.markdown("<hr style='border:1px solid #ccc;'>", unsafe_allow_html=True)

# CLEANING / ENCODING 
if 'MENSTRUAL' in df.columns and 'None' not in df['MENSTRUAL'].unique():
    df['MENSTRUAL'] = df['MENSTRUAL'].fillna('None')
    df.loc[df['GENDER'] == 'Male', 'MENSTRUAL'] = 'None'
st.subheader("🧼 ETL Pipeline")

# Encode categorical variables with individual encoders
encoders = {}
df_encoded = df.copy()
categorical_cols = df_encoded.select_dtypes(include='object').columns.drop("THYROID_TYPE")

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    encoders[col] = le

# Encode target separately
target_encoder = LabelEncoder()
df_encoded['THYROID_TYPE'] = target_encoder.fit_transform(df['THYROID_TYPE'])

st.success("Data encoded successfully!")

st.markdown("<hr style='border:1px solid #ccc;'>", unsafe_allow_html=True)
# EDA 
st.subheader("📊 Exploratory Data Analysis")
fig = px.histogram(df, x="THYROID_TYPE", color="THYROID_TYPE", title="Distribution of Thyroid Types")
st.plotly_chart(fig)

corr = df_encoded.corr()
fig2, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr, cmap="coolwarm", annot=True)
st.pyplot(fig2)

st.markdown("<hr style='border:1px solid #ccc;'>", unsafe_allow_html=True)
# MODEL TRAINING 
st.subheader("🧠 Model Training & Evaluation")
X = df_encoded.drop(["THYROID_TYPE", "BP"], axis=1)
y = df_encoded["THYROID_TYPE"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

st.text("Accuracy Score: {:.2f}".format(accuracy_score(y_test, y_pred)))
st.text("Precision Score: {:.2f}".format(precision_score(y_test, y_pred, average='weighted')))
st.text("Recall Score: {:.2f}".format(recall_score(y_test, y_pred, average='weighted')))
st.text("F1 Score: {:.2f}".format(f1_score(y_test, y_pred, average='weighted')))

st.text("\nClassification Report:")
st.code(classification_report(y_test, y_pred))

st.markdown("<hr style='border:1px solid #ccc;'>", unsafe_allow_html=True)
# USER INPUT 
st.subheader("🧾 Predict Your Thyroid Type")

def user_input_features():
    TSH = st.slider("TSH", 0.0, 100.0, 2.5)
    T3 = st.slider("T3", 0.0, 10.0, 1.5)
    T4 = st.slider("T4", 0.0, 15.0, 7.0)
    AGE = st.slider("AGE", 1, 100, 30)
    GENDER = st.selectbox("GENDER", ['Male', 'Female'])
    ANTIBODIES = st.selectbox("ANTITHYROID ANTIBODIES", ['Yes', 'No'])
    HEART_RATE = st.slider("HEART RATE", 40, 150, 70)
    CHOLESTEROL = st.slider("CHOLESTEROL", 100, 300, 180)
    BMI = st.slider("BMI", 10, 50, 24)
    BP = st.slider("BLOOD PRESSURE", 80, 180, 120)
    FAMILY = st.selectbox("FAMILY HISTORY", ['Yes', 'No'])
    GOITER = st.selectbox("GOITER", ['Yes', 'No'])
    GLUCOSE = st.slider("GLUCOSE", 60, 200, 90)
    VITD = st.slider("VIT D", 10, 100, 40)
    SMOKING = st.selectbox("SMOKING", ['Yes', 'No'])

    if GENDER == 'Female':
        MENSTRUAL = st.selectbox("MENSTRUAL", ['Irregular', 'Menopause'])
    else:
        MENSTRUAL = 'None'

    data = pd.DataFrame({
        'TSH': [TSH],
        'T3': [T3],
        'T4': [T4],
        'AGE': [AGE],
        'GENDER': [GENDER],
        'ANTITHYROID_ANTIBODIES': [ANTIBODIES],
        'HEART_RATE': [HEART_RATE],
        'CHOLESTEROL': [CHOLESTEROL],
        'BMI': [BMI],
        'BP': [BP],
        'MENSTRUAL': [MENSTRUAL],
        'FAMILY_HISTORY': [FAMILY],
        'GOITER': [GOITER],
        'GLUCOSE': [GLUCOSE],
        'VIT_D': [VITD],
        'SMOKING': [SMOKING]
    })
    return data

user_data = user_input_features()
st.write("### 👤 User Input Features", user_data)

st.markdown("<hr style='border:1px solid #ccc;'>", unsafe_allow_html=True)

user_data_encoded = user_data.copy()

if 'BP' in user_data_encoded.columns:
    user_data_encoded.drop(columns=['BP'], inplace=True)
    
if user_data_encoded.loc[0, "GENDER"] == "Male":
    user_data_encoded["MENSTRUAL"] = "None"
    
for col in user_data_encoded.columns:
    if col in encoders:
        try:
            user_data_encoded[col] = encoders[col].transform(user_data_encoded[col].astype(str))
        except ValueError as e:
            st.error(f"Unseen value '{user_data[col].values[0]}' in column '{col}'. Please check your input.")
            st.stop()
    else:
        # If column is numeric or not in encoders, keep as-is
        user_data_encoded[col] = pd.to_numeric(user_data_encoded[col], errors='coerce')

# ✅ Ensure all data is numeric
user_data_encoded = user_data_encoded.astype(float)

# 🔍 Debug (Optional)
# st.write("Encoded Data:", user_data_encoded)

# Scale and Predict
user_scaled = scaler.transform(user_data_encoded)
prediction = model.predict(user_scaled)
pred_class = target_encoder.inverse_transform(prediction)

st.success(f"🧬 Predicted Thyroid Condition: :rainbow[{pred_class[0]}]")

def generate_full_thyroid_report(name, email, user_data, predicted_condition):
    age = user_data["AGE"].values[0]
    gender = user_data["GENDER"].values[0]
    tsh = user_data["TSH"].values[0]
    t3 = user_data["T3"].values[0]
    t4 = user_data["T4"].values[0]
    cholesterol = user_data["CHOLESTEROL"].values[0]
    glucose = user_data["GLUCOSE"].values[0]
    bmi = user_data["BMI"].values[0]
    heart_rate = user_data["HEART_RATE"].values[0]
    vit_d = user_data["VIT_D"].values[0]
    menstrual = user_data["MENSTRUAL"].values[0] if "MENSTRUAL" in user_data.columns else "N/A"

    summary = f"""
👤 **Patient Name**: {name} \n
📧 **Email**: {email}\n
🧬 **Predicted Thyroid Condition**: {predicted_condition}

---

### 📊 **Entered Medical Values**

- **Age**: {age}
- **Gender**: {gender}
- **TSH**: {tsh} → {'High' if tsh > 4.0 else 'Low' if tsh < 0.4 else 'Normal'}
- **T3**: {t3}
- **T4**: {t4}
- **Cholesterol**: {cholesterol} → {'High' if cholesterol > 240 else 'Borderline' if cholesterol > 200 else 'Normal'}
- **Glucose**: {glucose} → {'Elevated' if glucose > 140 else 'Normal'}
- **BMI**: {bmi} → {'Overweight' if bmi >= 25 else 'Underweight' if bmi < 18.5 else 'Normal'}
- **Heart rate**: {heart_rate}
- **Vitamin D**: {vit_d}
- **Menstrual Status**: {menstrual}

---

### 🩺 **AI-Generated Doctor's Note**

"""

    # Doctor's note logic
    if predicted_condition == "Hypothyroid":
        summary += """
- Begin thyroid hormone replacement therapy as advised.
- Monitor weight and cholesterol levels regularly.
- Maintain a diet rich in iodine (e.g., dairy, seafood, iodized salt).
- Get regular thyroid function tests every 6–12 months.
"""
    elif predicted_condition == "Subclinical Hypothyroid":
        summary += """
- Condition is mild and may not present symptoms.
- Monitor TSH levels every 3–6 months to check progression.
- Discuss with your physician the need for low-dose hormone therapy.
- Maintain a balanced iodine intake.
"""
    elif predicted_condition == "Hyperthyroid":
        summary += """
- Antithyroid medication or radioactive iodine therapy might be required.
- Reduce iodine intake and avoid stimulants like caffeine and nicotine.
- Consider beta blockers for heart rate management if symptoms persist.
- Regular endocrine checkups are advised.
"""
    elif predicted_condition == "Subclinical Hyperthyroid":
        summary += """
- Mild or no symptoms, but may increase risk of heart arrhythmias or bone loss.
- Regularly monitor thyroid hormone levels every 3–6 months.
- Discuss cardiovascular and osteoporosis risk factors with your doctor.
- Treatment may not be immediately required but should be evaluated.
"""
    elif predicted_condition == "Thyroiditis":
        summary += """
- Anti-inflammatory medications may help with pain and swelling.
- Watch for fluctuations between hyperthyroid and hypothyroid phases.
- Recheck thyroid profile in 6–8 weeks.
- Seek medical help if symptoms worsen or persist.
"""
    elif predicted_condition == "Euthyroid":
        summary += """
- All values currently in the safe range. Keep up the healthy lifestyle.
- Recheck thyroid panel every 12 months, especially with family history or symptoms.
- Maintain a nutritious diet and manage stress levels.
"""
    else:
        summary += """
- Inconclusive results. Please consult with an endocrinologist for a detailed review.
- You may need more comprehensive lab tests for accurate diagnosis.
"""


    # Extra personal advice
    summary += "\n### 🔎 Additional AI Recommendations\n"

    if bmi > 30:
        summary += "- ⚠️ Your BMI indicates obesity. Weight management is strongly advised.\n"
    if cholesterol > 240:
        summary += "- ⚠️ High cholesterol detected. Consider dietary and lifestyle changes.\n"
    if vit_d < 30:
        summary += "- ⚠️ Low Vitamin D. Daily supplementation (1000–2000 IU) may be beneficial.\n"
    if heart_rate > 100:
        summary += "- ⚠️ Elevated heart rate. Monitor and consult a cardiologist if persistent.\n"

    summary += "\n🧠 _This is an AI-generated report. It does not substitute professional medical advice._"

    return summary



name = st.text_input("👤 Enter your Name (Required)", "")
email = st.text_input("📧 Enter your Email (Required)", "")

if not name or not email:
    st.warning("Please enter both your name and email to generate the full report.")
    st.stop()

if st.button("📄 Show Personalized Medical Report"):
    report = generate_full_thyroid_report(name, email, user_data, pred_class[0])
    with st.expander("🔍 View Detailed Medical Parameters"):
        st.balloons()
        st.write(report)



