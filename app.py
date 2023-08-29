import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px

# Load customer data
customer_data = pd.read_csv('assets/Mall_Customers.xls')
customer_data.drop('CustomerID', axis=1, inplace=True)

# Title
st.title("Customer Segmentation Analysis")

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(customer_data)

# Data preprocessing
df = customer_data.iloc[:, 1:]
scaler = StandardScaler()
dfs = scaler.fit_transform(df)

col1, col2 = st.columns([0.5,0.5])
# Elbow method
col1.subheader("Elbow Method for Optimal k")
inertia = []
for i in range(1, 11):
    kmeans = KMeans(init="k-means++", n_clusters=i, random_state=42)
    kmeans.fit(dfs)
    inertia.append(kmeans.inertia_)
plt.plot(range(1, 11), inertia, marker='o')
plt.title("No. of clusters and inertia")
plt.xlabel("Clusters")
plt.ylabel("Inertia")
col1.pyplot()

# Silhouette score
col2.subheader("Silhouette Score Method for Optimal k")
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(dfs)
    silhouette_scores.append(silhouette_score(dfs, labels))

plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score Method for Optimal k')
col2.pyplot()

# Cluster visualization
clusterNum = 6
kmeans = KMeans(n_clusters=clusterNum)
kmeans.fit(dfs)
labels = kmeans.labels_
slabel = labels.astype('str')

# Plot with Plotly Express
st.subheader("Clusters with Age by Income")
fig = px.scatter(data_frame=dfs, x=dfs[:, 0], y=dfs[:, 1], color_continuous_scale='darkmint', color=slabel,
                 labels={'size': 'cluster', 'x': 'Age', 'y': 'Annual Income (k$)', 'color': 'Cluster'},
                 hover_data={'Gender': customer_data['Gender'], 'Age': customer_data['Age'],
                             'Annual Income (k$)': customer_data['Annual Income (k$)']},
                 title='Clusters with Age by Income')

st.plotly_chart(fig, theme='streamlit', use_container_width=False)


# Function to predict cluster for user input
def Predict(INPUT):
    sample = np.array(INPUT)
    sample = sample.reshape(1, -1)
    result = kmeans.predict(sample)
    return result

# User input for prediction
st.subheader("Predict Cluster for User Input")
AGE = st.slider("Age:", 0, 100, 18)
INCOME = st.slider("Annual Income (k$):", 0, 200, 130)
SPENDING_SCORE = st.slider("Spending Score (1-100):", 0, 100, 45)

INPUT = [AGE, INCOME, SPENDING_SCORE]
result = Predict(INPUT)
st.write(f'Assigned Cluster: {result}')
