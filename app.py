import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from database import create_database, save_prediction, get_history


#load trained model and scaler
model=joblib.load("fraud_model.pkl")
scaler=joblib.load("scaler.pkl")
create_database()

#PAGE TITLE
st.title("Fraud Detection App")

st.write("Upload your CSV dataset below")
uploaded_file=st.file_uploader("UPload CSV File",type=["csv"])
if uploaded_file is not None:
    df=pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.write(df.head())
    #remove target column if present
    if "Class" in df.columns:
        input_data=df.drop("Class",axis=1)
    else:
        input_data=df
    #Scale data
    scaled_data=scaler.transform(input_data)
    st.write("✅ Scaling completed")
    #Prediction
    predictions=model.predict(scaled_data)
    st.write("✅ Prediction completed")
    #Add prediction column
    df["Prediction"]=predictions
    st.write(df.columns)
    # Save predictions to database
    for _, row in df.head(100).iterrows():
        amount = row["Amount"]
        result = "Fraud" if row["Prediction"] == 1 else "Non-Fraud"
        save_prediction(amount, result)
    #Show predictions
    st.subheader("Prediction Results")
    st.write(df.head())
    #Count fraud cases
    fraud_count=(predictions==1).sum()
    if fraud_count>0:
        st.error(f"{fraud_count} Fraud Transactions detected")
    else:
        st.success("No Fraud Transactions Found")
    #Accuracy (if actual label exists)
    if "Class" in df.columns:
        accuracy=accuracy_score(df["Class"],predictions)
        st.write(
            f"Model accuracy:{accuracy:.4f}"
        )
    #Show fraud rows only
    fraud_rows=df[df["Prediction"]==1]
    st.subheader("Fraud Transactions")
    st.dataframe(fraud_rows)
    #Download results
    csv=df.to_csv(index=False)
    st.download_button(label="Download Prediction Results",data=csv,
                        file_name="fraud_predictions.csv",mime="text/csv")
                        
# Prediction History
    st.subheader("Prediction History")

    history = get_history()

    st.write(f"Total saved predictions: {len(history)}")

    st.dataframe(history.tail(20))

   


    
