## Customer Segmentation (Unsupervised learning) v2
### <a href="https://cssegmentation.streamlit.app/">Streamlit Web app</a>
#### PRODIGY_ML_01
<hr>

### Pipeline / Workflow
Dataset -> EDA -> Extraction and cleaning -> Feature Engineering -> Full batch Modeling -> Clustering -> Evaluation

### Requirements
- Python 3.0 +
- Pandas
- Seaborn
- scikit-learn
- Matplotlib

### Design:
Received dataset (.csv) with labeled features to perform and execute Exploratory Analysis, proceeding with Modeling to segment/cluster groups of customers for developing insight with respect to features signifying patterns present in dataset, *which are subject to change in given time*.
For evaluation and picking the most efficient clusters, Elbow method and silhoutte score is performed.
The silhouette score for a single data point is then defined as (b - a) / max(a, b). The overall silhouette score for a set of data points is the average silhouette score across all points.
- a: The average distance from the data point to the other points in the same cluster.
- b: The smallest average distance from the data point to points in other clusters, minimizing over clusters.

Assets available to refer for further analysis are:
- <a href="https://www.kaggle.com/harshjadhav6301/mall-customer-segmentation">Jupyter notebook </a>
- Streamlit <a href="https://cssegmentation.streamlit.app/">web app</a> 
