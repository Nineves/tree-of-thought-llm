import numpy as np
from heapq import heapify, heappush, heappop
from itertools import permutations

def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def two_opt(route, distance_matrix):
    best = route
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1: continue  # changes nothing, skip then
                new_route = route[:]
                new_route[i:j] = route[j - 1:i - 1:-1]  # this is the 2-opt swap
                if total_distance(new_route, distance_matrix) < total_distance(best, distance_matrix):
                    best = new_route
                    improved = True
        route = best
    return best

def total_distance(route, distance_matrix):
    total = 0
    for i in range(len(route) - 1):
        total += distance_matrix[route[i]][route[i + 1]]
    total += distance_matrix[route[len(route) - 1]][route[0]]
    return total

def solve_tsp(points):
    n = len(points)
    # Precompute distance matrix
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance_matrix[i, j] = calculate_distance(points[i], points[j])

    # Generate an initial route using Nearest Neighbour
    start = points[0]
    must_visit = points
    path = [start]
    must_visit.remove(start)
    while must_visit:
        nearest = min(must_visit, key=lambda x: calculate_distance(path[-1], x))
        path.append(nearest)
        must_visit.remove(nearest)

    # Convert path to indices
    path_indices = [points.index(point) for point in path]

    # Optimize the initial path using 2-opt
    optimized_path = two_opt(path_indices, distance_matrix)

    return optimized_path