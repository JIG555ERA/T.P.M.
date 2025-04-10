import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

# setting the page configurations
st.set_page_config(
    page_title="Analyze | T.P.M.",
    # page_icon="./tpm_icon.png"
)

# st.sidebar.image("company_icon.png", use_column_width=True)

# 1. Page Title and Description
df = pd.read_csv("./thyroid_dataset.csv")
df_encoded = df.copy()
df_encoded["THYROID_TYPE_ENCODED"] = LabelEncoder().fit_transform(df["THYROID_TYPE"])
numeric_df = df_encoded.select_dtypes(include=['float64', 'int64'])

st.title("📊 Thyroid Data Analysis")
st.markdown("""
Explore thyroid health patterns and compare key medical features across different thyroid conditions.
Use the visualizations below to gain a better understanding of the dataset.
""")

st.markdown("<hr style='border:1px solid #ccc;'>", unsafe_allow_html=True)
# 2. Thyroid Type Distribution
st.subheader("🔢 Distribution of Thyroid Conditions")
fig1 = px.histogram(df, x="THYROID_TYPE", color="THYROID_TYPE", title="Number of Patients per Thyroid Type")
st.plotly_chart(fig1)

st.markdown("<hr style='border:1px solid #ccc;'>", unsafe_allow_html=True)
# 3. Hormone Levels by Thyroid Type (Box Plots)
st.subheader("🧪 Hormone Levels Across Conditions")

features = ["TSH", "T3", "T4"]
for feature in features:
    fig = plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x="THYROID_TYPE", y=feature, palette="pastel")
    plt.title(f"{feature} by Thyroid Type")
    st.pyplot(fig)
    st.markdown("<hr style='border:1px solid #ccc;'>", unsafe_allow_html=True)

# st.markdown("<hr style='border:1px solid #ccc;'>", unsafe_allow_html=True)
# 4. Feature Correlation Heatmap
# st.subheader("📌 Feature Correlation")

# fig = plt.figure(figsize=(12, 8))
# sns.heatmap(df.drop(columns=["THYROID_TYPE"]).corr(), annot=True, cmap="coolwarm", fmt=".2f")
# plt.title("Correlation Heatmap of Medical Features")
# st.pyplot(fig)

# 5. Compare Any Two Features (Interactive Scatter Plot)
st.subheader("🎯 Compare Two Features")

col1, col2 = st.columns(2)
x_axis = col1.selectbox("X-axis Feature", df.columns[:-1])
y_axis = col2.selectbox("Y-axis Feature", df.columns[:-1])

fig2 = px.scatter(df, x=x_axis, y=y_axis, color="THYROID_TYPE", title=f"{x_axis} vs {y_axis}")
st.plotly_chart(fig2)

st.markdown("<hr style='border:1px solid #ccc;'>", unsafe_allow_html=True)
# 6. Distribution of Numeric Features (Histograms)
st.subheader("📈 Distribution of Numeric Features")

selected_feat = st.selectbox("Choose a feature to visualize:", df.select_dtypes(include=["float64", "int"]).columns)
fig3 = px.histogram(df, x=selected_feat, color="THYROID_TYPE", nbins=30, barmode="overlay")
st.plotly_chart(fig3)

st.markdown("<hr style='border:1px solid #ccc;'>", unsafe_allow_html=True)
# 7. Violin Plot – Cholesterol or Heart Rate
st.subheader("🎻 Violin Plot for Feature Distributions")

selected_violin = st.selectbox("Choose a feature (like HEART_RATE or CHOLESTEROL):", ["HEART_RATE", "CHOLESTEROL", "BMI"])
fig4 = px.violin(df, y=selected_violin, x="THYROID_TYPE", box=True, color="THYROID_TYPE", title=f"{selected_violin} Distribution by Thyroid Type")
st.plotly_chart(fig4)

st.markdown("<hr style='border:1px solid #ccc;'>", unsafe_allow_html=True)
# 8. Pair Plot (Optional – Static Overview of Key Features)
st.subheader("🔗 Pairwise Plot of Core Features (TSH, T3, T4)")

sample_df = df[["TSH", "T3", "T4", "THYROID_TYPE"]].sample(500)  # To avoid slow rendering
sns.set(style="ticks")
fig5 = sns.pairplot(sample_df, hue="THYROID_TYPE", palette="Set2")
st.pyplot(fig5)

st.markdown("<hr style='border:1px solid #ccc;'>", unsafe_allow_html=True)
# 9. Extra: Feature Averages per Condition (Bar Plot)
st.subheader("📊 Average Feature Values per Thyroid Type")

avg_df = df.groupby("THYROID_TYPE")[["TSH", "T3", "T4", "HEART_RATE", "CHOLESTEROL", "BMI"]].mean().reset_index()
fig6 = px.bar(avg_df.melt(id_vars="THYROID_TYPE"), x="variable", y="value", color="THYROID_TYPE", barmode="group")
st.plotly_chart(fig6)

st.markdown("### 📘 Chart Descriptions & Insights")

chart_descriptions = {
    "1. Thyroid Type Distribution":
        "This chart shows the number of records for each thyroid type. It gives an overview of how balanced the dataset is, which is important for training a model.",
    
    "2. Hormone Levels by Thyroid Type (Box Plots)":
        "These box plots show how TSH, T3, and T4 levels vary across different thyroid conditions. For example, Hypothyroidism typically shows high TSH and low T3/T4, while Hyperthyroidism shows the opposite.",

    "3. Feature Correlation Heatmap":
        "This heatmap displays how features are linearly correlated. It helps identify redundancy or strong relationships, such as between BMI and cholesterol, or between glucose and age.",
    
    "4. Compare Any Two Features (Scatter Plot)":
        "This interactive scatter plot allows you to compare two selected features. It helps identify clusters, outliers, or separation between classes.",
    
    "5. Distribution of Numeric Features (Histogram)":
        "These histograms show how a single numeric feature is distributed across different thyroid types. It helps spot skewed or bimodal distributions.",
    
    "6. Violin Plot for Feature Distributions":
        "Violin plots combine box plots and KDEs to show distribution and density. They highlight the shape and spread of features like heart rate or cholesterol across thyroid conditions.",
    
    "7. Pairwise Plot of Core Features":
        "Pair plots give a comprehensive overview of how several features interact with each other. They're great for spotting patterns, correlations, and class separability.",
    
    "8. Average Feature Values per Condition":
        "This bar plot compares mean values of selected features grouped by thyroid condition. It clearly shows how certain features (like TSH or heart rate) differ between conditions.",
    
    "9. Pie Chart – Proportion of Each Thyroid Condition":
        "The pie chart visually represents the share of each thyroid condition in the dataset. It’s a simple way to show whether the data is imbalanced or balanced."
}

selected_chart = st.selectbox("📋 Select a chart to view its description:", list(chart_descriptions.keys()))
st.info(chart_descriptions[selected_chart])
