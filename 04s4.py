import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Sample data
data = pd.DataFrame({
    'X': np.random.randn(100),
    'Y': np.random.randn(100),
    'Z': np.random.randn(100)
})

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Apply PCA
pca = PCA(n_components=2)
pca_components = pca.fit_transform(data_scaled)

# Create a Streamlit interface
st.title("PCA Projection")

# Scatter plot of PCA components
fig, ax = plt.subplots()
ax.scatter(pca_components[:, 0], pca_components[:, 1], color='grey')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
st.pyplot(fig)

# Display projection axes
fig, ax = plt.subplots()
ax.quiver(0, 0, pca.components_[0, 0], pca.components_[1, 0], angles='xy', scale_units='xy', scale=1, color="red", label="PC1")
ax.quiver(0, 0, pca.components_[0, 1], pca.components_[1, 1], angles='xy', scale_units='xy', scale=1, color="blue", label="PC2")
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.legend()
st.pyplot(fig)
