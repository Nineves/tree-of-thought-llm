import numpy as np
from heapq import heapify, heappush, heappop

def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def solve_tsp(points):
    n = len(points)

    # Precompute distance matrix
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance_matrix[i, j] = calculate_distance(points[i], points[j])

    # Create a set to track points in the path
    in_path = set()
    # Start from the first point
    in_path.add(0)
    tsp_path = [0]

    # Initialize priority queue
    pq = []

    for point in range(1, n):
        increment = distance_matrix[0][point]
        heappush(pq, (increment, point, 0))

    while pq:
        increment, point, i = heappop(pq)
        if point in in_path: continue

        in_path.add(point)
        tsp_path.insert(i+1, point)

        for next_point in range(0, n):
            if next_point not in in_path:
                if i+1 == len(tsp_path) - 1:
                    new_increment = distance_matrix[tsp_path[i+1]][next_point] + distance_matrix[next_point][tsp_path[0]] - distance_matrix[tsp_path[i+1]][tsp_path[0]]
                else:
                    new_increment = distance_matrix[tsp_path[i+1]][next_point] + distance_matrix[next_point][tsp_path[i+2]] - distance_matrix[tsp_path[i+1]][tsp_path[i+2]]
                heappush(pq, (new_increment, next_point, i+1))
    return tsp_path