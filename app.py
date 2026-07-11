import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from database import create_database, save_prediction, get_history
import plotly.express as px


# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="AI Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)


#load trained model and scaler
model=joblib.load("fraud_model.pkl")
scaler=joblib.load("scaler.pkl")
create_database()

# ---------------- SIDEBAR ---------------- #
with st.sidebar:
    st.title("💳 Fraud Detection")

    st.markdown("---")

    st.write("### Navigation")

    page = st.radio(
        "",
        [
            "🏠 Dashboard",
            "📜 Prediction History",
            "ℹ About"
        ]
    )

    st.markdown("---")

    st.success("🤖 Model: Random Forest")

    st.info("🎯 F1 Score: 76.1%")
    st.info("📈 ROC-AUC: 97.9%")

#PAGE TITLE
if page == "🏠 Dashboard":
    st.title("💳 AI Fraud Detection Dashboard")

    st.caption(
        "Detect fraudulent financial transactions using Machine Learning."
    )

    st.markdown("---")

    st.subheader("📂 Upload Transaction Dataset")

    uploaded_file=st.file_uploader("UPload CSV File",type=["csv"])
    if uploaded_file is not None:
        df=pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(
        df.head(),
        use_container_width=True
    )
    #remove target column if present
        if "Class" in df.columns:
            input_data=df.drop("Class",axis=1)
        else:
            input_data=df
    #Scale data
        with st.spinner("Analyzing transactions..."):

            scaled_data = scaler.transform(input_data)

            predictions = model.predict(scaled_data)

            probabilities = model.predict_proba(scaled_data)[:, 1]
                
   
    #Add prediction column
        df["Prediction"]=predictions
        df["Fraud Probability"] = (probabilities * 100).round(2)

        def risk_level(prob):

            if prob >= 80:
                return "🔴 High"

            elif prob >= 40:
                return "🟠 Medium"

            else:
                return "🟢 Low"

        df["Risk Level"] = df["Fraud Probability"].apply(risk_level)

        total_transactions = len(df)
        fraud_count = (predictions == 1).sum()
        nonfraud_count = total_transactions - fraud_count
        fraud_rate = (fraud_count / total_transactions) * 100

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "📄 Total Transactions",
            total_transactions
        )

        col2.metric(
            "🚨 Frauds",
            fraud_count
        )

        col3.metric(
            "✅ Legitimate",
            nonfraud_count
        )

        col4.metric(
            "📈 Fraud %",
            f"{fraud_rate:.2f}%"
        )

        st.markdown("---")

        st.subheader("📊 Transaction Overview")

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:

            pie_df = pd.DataFrame({
                "Type": ["Legitimate", "Fraud"],
                "Count": [nonfraud_count, fraud_count]
            })

            fig = px.pie(
                pie_df,
                names="Type",
                values="Count",
                title="Fraud Distribution",
                hole=0.45
            )

            st.plotly_chart(fig, use_container_width=True)

        with chart_col2:

            pred_counts = df["Prediction"].replace({
                0: "Legitimate",
                1: "Fraud"
            }).value_counts()

            fig = px.bar(
                x=pred_counts.index,
                y=pred_counts.values,
                labels={
                    "x": "Prediction",
                    "y": "Count"
                },
                title="Prediction Counts"
            )

            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("💰 Transaction Amount Distribution")

        fig = px.histogram(
            df,
            x="Amount",
            nbins=50,
            title="Transaction Amount Distribution"
        )
    
        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.markdown("---")
        st.subheader("📦 Fraud vs Transaction Amount")

        plot_df = df.copy()

        plot_df["Prediction"] = plot_df["Prediction"].replace({
            0: "Legitimate",
            1: "Fraud"
        })

        fig = px.box(
            plot_df,
            x="Prediction",
            y="Amount",
            color="Prediction",
            title="Transaction Amount Comparison"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # st.write(df.columns)
    # Save predictions to database
        for _, row in df.head(100).iterrows():
            amount = row["Amount"]
            result = "Fraud" if row["Prediction"] == 1 else "Non-Fraud"
            save_prediction(
                amount,
                result,
                row["Fraud Probability"],
                row["Risk Level"]
            )
    #Show predictions
        with st.expander("📊 View Prediction Results"):
            st.dataframe(
                df,
                use_container_width=True
            )
    #Count fraud cases
        fraud_count=(predictions==1).sum()
        if fraud_count > 0:
            st.error(f"🚨 High Risk: {fraud_count} fraudulent transactions detected.")
        else:
            st.success("✅ No fraudulent transactions detected.")
    #Accuracy (if actual label exists)
        if "Class" in df.columns:
            accuracy=accuracy_score(df["Class"],predictions)
            st.info(f"🎯 Model Accuracy: {accuracy:.4f}")
    #Show fraud rows only
        fraud_rows=df[df["Prediction"]==1]
        st.markdown("---")
        st.subheader("💸 Highest Fraud Amounts")

        if not fraud_rows.empty:

            top_fraud = fraud_rows.sort_values(
                by="Amount",
                ascending=False
            ).head(10)

            fig = px.bar(
                top_fraud,
                x=top_fraud.index.astype(str),
                y="Amount",
                title="Top Fraudulent Transactions"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )
    
        with st.expander("🚨 View Fraud Transactions"):

            st.dataframe(
                fraud_rows[
                    [
                        "Amount",
                        "Fraud Probability",
                        "Risk Level",
                        "Prediction"
                    ]
                ],
                use_container_width=True
            )
    #Download results
        csv=df.to_csv(index=False)
        st.download_button(label="⬇ Download Prediction Results",data=csv,
                        file_name="fraud_predictions.csv",mime="text/csv")
                        
# Prediction History
elif page == "📜 Prediction History":

    st.title("📜 Prediction History")

    history = get_history()

    st.metric(
        "Total Predictions Saved",
        len(history)
    )

    st.dataframe(
        history,
        use_container_width=True
    ) 

elif page == "ℹ About":

    st.title(" About This Project")

    st.markdown("""
# 💳 AI Fraud Detection System

## Project Overview

This project detects fraudulent financial transactions using Machine Learning.

The application analyses uploaded transaction datasets, predicts fraud probability, assigns a risk level, stores predictions in SQLite, and visualizes results using an interactive Streamlit dashboard.

---

## Dataset

• European Credit Card Fraud Dataset

• 284,807 transactions

• Highly imbalanced (0.17% fraud)

---

## Machine Learning Pipeline

✔ Data Cleaning

✔ Duplicate Removal

✔ Standard Scaling

✔ SMOTE Oversampling

✔ Model Training

✔ Risk Scoring

---

## Final Model

Random Forest Classifier

---

## Performance

Accuracy : 99.92%

Precision : 73.5%

Recall : 78.9%

F1 Score : 76.1%

ROC-AUC : 97.9%

---

## Technologies

• Python

• Streamlit

• Scikit-learn

• Plotly

• SQLite

• FastAPI

• Joblib
""")
    
st.markdown("---")

st.caption(
    "Developed using Streamlit • Scikit-learn • FastAPI • SQLite"
)