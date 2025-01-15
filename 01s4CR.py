import streamlit as st
import pandas as pd

st.title("McDonald's Dataset")


#Loading the Dataset
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if  uploaded_file is not None:
    mcdonalds = pd.read_csv(r"C:\Users\AMAAN\OneDrive\Desktop\ML internship\T-1\step4CR\csv\mcdonalds.csv")

    #View Column Names
    st.subheader("column Names:")
    st.write(mcdonalds.columns.tolist())

    #Checking Dataset in Dimension
    st.subheader("\nDataset Dimenions:")
    st.write(f"Rows: {mcdonalds.shape[0]}, Columns: {mcdonalds.shape[1]}")

    #View the first 3 rowa of the dataset

    st.subheader("First 3 Rows of the Dataset")
    st.write(mcdonalds.head(3))
else:
    st.write("No file uploaded.Please upload CVS file to viw the dataset")