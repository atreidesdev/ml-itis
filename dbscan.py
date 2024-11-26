import pygame
import numpy as np

pygame.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('DBSCAN с Pygame')

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

points = []
flags = []
clusters = []
visited = []
core_points = []

eps = 100
min_samples = 5

flag_colors = [BLUE, RED, GREEN]
current_flag_idx = 0

def distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def region_query(point_idx):
    neighbors = []
    for idx, point in enumerate(points):
        if distance(points[point_idx], point) <= eps and flags[point_idx] == flags[idx]:
            neighbors.append(idx)
    return neighbors

def expand_cluster(point_idx, neighbors, cluster_idx):
    clusters[point_idx] = cluster_idx
    i = 0
    while i < len(neighbors):
        neighbor_idx = neighbors[i]
        if not visited[neighbor_idx]:
            visited[neighbor_idx] = True
            new_neighbors = region_query(neighbor_idx)
            if len(new_neighbors) >= min_samples:
                neighbors += new_neighbors
        if clusters[neighbor_idx] == -1:
            clusters[neighbor_idx] = cluster_idx
        i += 1

def dbscan():
    global clusters, visited, core_points
    clusters = [-1] * len(points)
    visited = [False] * len(points)
    core_points = []
    cluster_idx = 0

    for i in range(len(points)):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = region_query(i)
        if len(neighbors) < min_samples:
            clusters[i] = -1
        else:
            clusters[i] = cluster_idx
            core_points.append(i)
            expand_cluster(i, neighbors, cluster_idx)
            cluster_idx += 1

def generate_random_color():
    return (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

def draw_clusters():
    screen.fill(WHITE)
    cluster_colors = {}
    for i, point in enumerate(points):
        if clusters[i] == -1:
            color = BLACK
        else:
            cluster_idx = clusters[i]
            if cluster_idx not in cluster_colors:
                cluster_colors[cluster_idx] = generate_random_color()
            color = cluster_colors[cluster_idx]
        pygame.draw.circle(screen, color, (int(point[0]), int(point[1])), 5)
    pygame.display.flip()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                points.append(np.array([event.pos[0], event.pos[1]]))
                flags.append(flag_colors[current_flag_idx])
            screen.fill(WHITE)
            for i, point in enumerate(points):
                pygame.draw.circle(screen, flags[i], (int(point[0]), int(point[1])), 5)
            pygame.display.flip()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                current_flag_idx = (current_flag_idx + 1) % len(flag_colors)
            elif event.key == pygame.K_r:
                dbscan()
                draw_clusters()

pygame.quit()
