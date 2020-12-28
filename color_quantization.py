import cv2
import numpy as np
from matplotlib import pyplot as plt


# Plot - Scatter
def scatter_plot(X, centroids, cluster_assigned, fig_name):
    # Ploting the red points.
    plt.scatter(centroids[:, 0][0], centroids[:, 1][0], s=50, facecolor='red',
                edgecolors="red", marker='o')
    plt.text(centroids[:, 0][0], centroids[:, 1][0],
             s=" ({:.1f},{:.1f})".format(centroids[:, 0][0], centroids[:, 1][0]),
             horizontalalignment='left',
             verticalalignment='top', fontsize=8)
    red_points = X[np.where(cluster_assigned == 0)]
    plt.scatter(red_points[:, 0], red_points[:, 1], facecolor='red',
                edgecolors="red", marker='^')
    for point in red_points:
        plt.text(point[0], point[1], s=" ({:.1f},{:.1f})".format(point[0], point[1]),
                 horizontalalignment='left', verticalalignment='top',
                 fontsize=8)

    # Ploting the green points.
    plt.scatter(centroids[:, 0][1], centroids[:, 1][1], s=50, facecolor='green',
                edgecolors="green", marker='o')
    plt.text(centroids[:, 0][1], centroids[:, 1][1],
             s=" ({:.1f},{:.1f})".format(centroids[:, 0][1], centroids[:, 1][1]),
             horizontalalignment='left', verticalalignment='top',
             fontsize=8)
    green_points = X[np.where(cluster_assigned == 1)]
    plt.scatter(green_points[:, 0], green_points[:, 1], facecolor='green',
                edgecolors="green", marker='^')
    for point in green_points:
        plt.text(point[0], point[1], s=" ({:.1f},{:.1f})".format(point[0], point[1]),
                 horizontalalignment='left', verticalalignment='top',
                 fontsize=8)

    # Ploting the blue points.
    plt.scatter(centroids[:, 0][2], centroids[:, 1][2], s=50, facecolor='blue',
                edgecolors="blue", marker='o')
    plt.text(centroids[:, 0][2], centroids[:, 1][2],
             s=" ({:.1f},{:.1f})".format(centroids[:, 0][2], centroids[:, 1][2]),
             horizontalalignment='left', verticalalignment='top', fontsize=8)
    blue_points = X[np.where(cluster_assigned == 2)]
    for point in blue_points:
        plt.text(point[0], point[1], s=" ({:.1f},{:.1f})".format(point[0], point[1]),
                 horizontalalignment='left', verticalalignment='top', fontsize=8)
        plt.scatter(blue_points[:, 0], blue_points[:, 1], facecolor='blue',
                    edgecolors="blue", marker='^')
    plt.title(fig_name.replace(".jpg", ""))
    plt.savefig(fig_name)
    plt.close()


# Computes the distance from each centroid to all data points.
def calculate_distance(points, centroids):
    return np.linalg.norm(np.array([points - np.ones(points.shape) * centroid for centroid in centroids]), axis=(2))


# Custom kMean Clustering function.
def kMeans(X, k, centroids, maxIter, plot):
    for i in range(maxIter):
        distances = calculate_distance(X, centroids)
        cluster_assigned = np.argmin(distances, axis=0)
        # Scatter Plot
        if plot == "scatter":
            fig_name = "task2_iter{}_a.jpg".format(i + 1)
            scatter_plot(X, centroids, cluster_assigned, fig_name)
        # Recomputing the centroids based on assigned clusters
        centroids = np.array(np.array([X[cluster_assigned == k].mean(axis=0) for k in range(k)]))
        # Scatter Plot
        if plot == "scatter":
            fig_name = "task2_iter{}_b.jpg".format(i + 1)
            scatter_plot(X, centroids, cluster_assigned, fig_name)

        if plot == "color_quantization":
            image_name = 'task2_baboon_{}.jpg'.format(k)
            color_quantization_plots(cluster_assigned, centroids, image_name)

    return cluster_assigned, centroids


# Plot - Color Quantization
def color_quantization_plots(assigned_clusters, centroids, image_name):
    centroids = np.uint8(centroids)
    output = centroids[assigned_clusters.flatten()]
    output = output.reshape(image.shape)
    cv2.imwrite(image_name, output)


# Task 2 - kMeans
X = np.array(
    [[5.9, 3.2], [4.6, 2.9], [6.2, 2.8], [4.7, 3.2], [5.5, 4.2], [5.0, 3.0], [4.9, 3.1], [6.7, 3.1], [5.1, 3.8],
     [6.0, 3.0]])
initial_centriods = np.array([[6.2, 3.2], [6.6, 3.7], [6.5, 3.0]])
k = 3
assigned_clusters, centroids = kMeans(X, k, initial_centriods, maxIter=2, plot="scatter")

# Task 2 - Colour Quantization
image = cv2.imread("baboon.png")
image_reshaped = np.float32(image).reshape((-1, 3))
k = [3, 5, 10, 20]

for i in range(len(k)):
    initial_centriods = image_reshaped[np.random.choice(np.arange(len(image_reshaped)), k[i])]
    assigned_clusters, centroids = kMeans(image_reshaped, k[i], initial_centriods, maxIter=10,
                                          plot="color_quantization")


