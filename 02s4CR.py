import streamlit as st
import pandas as pd
import numpy as np

st.title("Numeric Segmentation")

#Loading the Dataset
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file is not None:
    mcdonalds = pd.read_csv(r"C:\Users\AMAAN\OneDrive\Desktop\ML internship\T-1\step4CR\csv\mcdonalds.csv")
    
    # Extracting First 11 Coloumns
    MD_x = mcdonalds.iloc[:, 0:11]
    
    # Convert "Yes" to1 and other values into 0
    MD_x = MD_x.applymap(lambda x: 1 if x == "Yes" else 0)
    
    # Calculate column means and round to 2 decimal places
    column_means = MD_x.mean(axis=0).round(2)
    
    # Display the results
    st.subheader("Column Means (Propotion of 'Yes')")
    st.write(column_means)
    
    # Let the user choose the number of rows to display
    st.subheader("View Dataset")
    num_rows = st.slider("Select the number of rows to display :" , min_value= 1, max_value= len(mcdonalds), value= 5)
    
    st.write(f"Displaying the first {num_rows} rows of the dataset:")
    st.dataframe(mcdonalds.head(num_rows)) 
    
    # # Optionally show the original data
    # if st.checkbox("Show First 10 Rows of the Dataset"):
    #     st.write(mcdonalds.head())
    # else:
    #     st.info("Please upload CSV File to Proceed")
    
