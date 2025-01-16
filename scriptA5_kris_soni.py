from kmodes.kmodes import KModes
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'C:/Users/Kris/Dropbox/PC/Downloads/car+evaluation/car.data'  # Replace with the actual file path
columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
car_data = pd.read_csv(file_path, header=None, names=columns)

# Prepare the data for K-Modes clustering (exclude target variable 'class')
data_for_clustering = car_data.iloc[:, :-1]  # Exclude the target column

# Initialize and fit the K-Modes model
n_clusters = 8  # Replace with the desired number of clusters
km = KModes(n_clusters=n_clusters, init='Huang', n_init=10, verbose=1)

# Fit the model and predict cluster labels
clusters = km.fit_predict(data_for_clustering)

# Add cluster labels to the original dataset
car_data['cluster'] = clusters

# Analyze cluster counts
print("Cluster distribution:")
print(car_data['cluster'].value_counts())

# Save the clustered dataset to a CSV file
car_data.to_csv('kmodes_clusters.csv', index=False)

# Print cluster centroids
print("Cluster centroids:")
print(km.cluster_centroids_)

# Visualize the clustering
sns.countplot(x='cluster', data=car_data)
plt.title('Cluster Distribution')
plt.xlabel('Cluster')
plt.ylabel('Number of Instances')
plt.show()

# For better visualization, you can display car names in the cluster plot:
# Assuming you have a column 'car_name' (this would be from car_names.csv or a similar file)
# Here we just show an example using the top 5 car names in each cluster:

for cluster_num in car_data['cluster'].unique():
    cluster_data = car_data[car_data['cluster'] == cluster_num]
    # If car names are available, extract the top 5 car names
    # car_names_in_cluster = ', '.join(cluster_data['car_name'].head(5))
    # If no car names, just show the first few rows
    car_names_in_cluster = ', '.join(cluster_data['buying'].head(5))  # You can change this to a meaningful column
    print(f"Cluster {cluster_num}: {car_names_in_cluster}")

# Optional: Save the summary of clusters to a file
cluster_summary = car_data.groupby('cluster').agg({
    'buying': pd.Series.mode,
    'maint': pd.Series.mode,
    'doors': pd.Series.mode,
    'persons': pd.Series.mode,
    'lug_boot': pd.Series.mode,
    'safety': pd.Series.mode
}).reset_index()

print("\nCluster Summary:")
print(cluster_summary)
