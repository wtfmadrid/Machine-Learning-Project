# Machine-Learning-Project

# Overview
This project demonstrates the use of the K-Modes clustering algorithm to group car evaluation data into distinct clusters based on categorical features. The dataset used for this project is the "Car Evaluation Dataset," which evaluates cars based on multiple categorical attributes like buying price, maintenance cost, and safety level. The primary goal is to identify patterns and create meaningful groupings of cars to support decision-making.

# Features
K-Modes Clustering: Implementation of the K-Modes algorithm for clustering categorical data.
Dataset Analysis: Preprocessing and exploration of the car evaluation dataset.
Cluster Analysis: Evaluation of clusters, including their size distribution and cluster centroids.
Visualization: Visual representation of cluster distributions using Seaborn and Matplotlib.
Cluster Insights: Summary of the most frequent attributes within each cluster.

# Dataset

The dataset contains the following columns:

buying: Buying price (categorical)
maint: Maintenance cost (categorical)
doors: Number of doors (categorical)
persons: Number of persons it can accommodate (categorical)
lug_boot: Size of luggage boot (categorical)
safety: Safety level (categorical)
class: Overall car acceptability (excluded from clustering)

# Requirements

To run the project, the following libraries are required:

Python 3.6+
pandas
matplotlib
seaborn
scikit-learn
kmodes

# Steps to Reproduce

Dataset Preparation:

Load the dataset car.data into a pandas DataFrame.
Ensure the dataset contains all the required columns as described above.
K-Modes Clustering:

Use the KModes module from kmodes to initialize and fit the clustering model.
Specify the number of clusters (n_clusters) and other hyperparameters.
Cluster Assignment:

Predict the cluster labels for each data point and append them to the dataset.
Visualization:

Use Seaborn's countplot to visualize the distribution of instances across clusters.
Cluster Insights:

Analyze the cluster centroids to understand the common features of each cluster.
Save the cluster summary to a CSV file for further analysis.

# Results:

View the distribution of instances across clusters.
Identify key patterns and groupings within the data.
Output
Clustered Dataset: The final dataset, including the cluster assignments, is saved as kmodes_clusters.csv.
Cluster Centroids: Displayed in the console to understand the characteristics of each cluster.
Visualization: Bar chart of cluster distributions.
