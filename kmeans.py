import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
import imageio

iris = load_iris()
data = iris.data

inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.title("Метод локтя для выбора оптимального k")
plt.xlabel("Количество кластеров")
plt.grid()
plt.show()

optimal_k = 3

def initialize_centroids(data, k):
    indices = np.random.choice(data.shape[0], k, replace=False)
    return data[indices]

def assign_clusters(data, centroids):
    return pairwise_distances_argmin(data, centroids)

def update_centroids(data, labels, k):
    new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
    return new_centroids

def plot_clusters(data, labels, centroids, step, save_path):
    plt.figure(figsize=(8, 5))
    for i, color in enumerate(['r', 'g', 'b']):
        points = data[labels == i]
        plt.scatter(points[:, 0], points[:, 1], s=30, color=color, label=f'Кластер {i+1}')
    plt.scatter(centroids[:, 0], centroids[:, 1], s=200, color='yellow', marker='X', label='Центроиды')
    plt.title(f"Шаг {step}")
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    plt.close()

def kmeans_manual(data, k, max_iter=100):
    centroids = initialize_centroids(data, k)
    prev_centroids = centroids + 10
    step = 0
    images = []
    while not np.allclose(centroids, prev_centroids) and step < max_iter:
        step += 1
        prev_centroids = centroids
        labels = assign_clusters(data, centroids)
        centroids = update_centroids(data, labels, k)

        save_path = f"step_{step}.png"
        plot_clusters(data, labels, centroids, step, save_path)
        images.append(save_path)

    return labels, centroids, images

np.random.seed(42)
labels, centroids, image_paths = kmeans_manual(data, optimal_k)

with imageio.get_writer("kmeans_steps.gif", mode="I", duration=1) as writer:
    for image_path in image_paths:
        writer.append_data(imageio.imread(image_path))
