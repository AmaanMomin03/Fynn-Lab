import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

# Streamlit app title
st.title("Principal Component Analysis (PCA) Analysis")

# File upload for CSV data
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

if uploaded_file is not None:
    # Load the data
    MD_x = pd.read_csv(uploaded_file)
    
    # Display data preview
    st.write("Data Preview", MD_x.head())

    # Check for non-numeric columns
    non_numeric_cols = MD_x.select_dtypes(exclude=[np.number]).columns
    st.write("Non-numeric columns:", non_numeric_cols)

    # Convert categorical columns to numeric
    label_encoder = LabelEncoder()
    for col in non_numeric_cols:
        MD_x[col] = label_encoder.fit_transform(MD_x[col])
    
    # Check for missing values and handle them (e.g., fill with column mean)
    if MD_x.isnull().any().any():
        st.write("There are missing values. Filling missing values with the column mean.")
        MD_x = MD_x.fillna(MD_x.mean())
    
    # Standardizing the data
    st.subheader("Standardizing the Data")
    scaler = StandardScaler()
    MD_x_scaled = scaler.fit_transform(MD_x)

    # Perform PCA
    pca = PCA()
    MD_pca = pca.fit(MD_x_scaled)

    # Explained Variance and Cumulative Variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = explained_variance.cumsum()

    # Show explained variance in Streamlit
    st.subheader("Explained Variance Ratio")
    st.write(explained_variance)

    st.subheader("Cumulative Explained Variance Ratio")
    st.write(cumulative_variance)

    # Plot Explained Variance and Cumulative Variance
    st.subheader("Explained Variance Plot")
    fig, ax = plt.subplots()
    ax.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', label='Variance Explained')
    ax.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', label='Cumulative Variance')
    ax.set_xlabel('Principal Components')
    ax.set_ylabel('Variance')
    ax.legend()
    st.pyplot(fig)

    # Factor Loadings (Components)
    factor_loadings = pd.DataFrame(pca.components_, columns=MD_x.columns)

    # Display Factor Loadings
    st.subheader("Factor Loadings (Components)")
    st.write(factor_loadings)

    # Show number of components to keep (Optional)
    num_components = st.slider("Select number of principal components", min_value=1, max_value=len(explained_variance), value=2)
    st.write(f"Number of Components selected: {num_components}")
    
    # Plotting the principal components (Optional)
    if num_components == 2:
        pca_2d = pca.transform(MD_x_scaled)[:, :2]
        st.subheader("2D PCA Projection")
        st.write("Displaying the projection of the first two principal components.")
        fig, ax = plt.subplots()
        ax.scatter(pca_2d[:, 0], pca_2d[:, 1])
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        st.pyplot(fig)
