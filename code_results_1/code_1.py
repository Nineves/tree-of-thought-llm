import itertools
import math

# First, calculate the Euclidean distance between two points
def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def solve_tsp(points):
    distances = []
    for i in range(len(points)):
        row = []
        for j in range(len(points)):
            row.append(calculate_distance(points[i], points[j]))
        distances.append(row)

    # Initialize a random tour
    tour = list(range(len(points)))
    tour.append(tour[0])

    for i in range(len(tour) - 1):
        min_distance = float('inf')
        min_index = -1
        for j in range(i + 2, len(tour)):
            old_distance = distances[tour[i]][tour[i + 1]] + distances[tour[j - 1]][tour[j]]
            new_distance = distances[tour[i]][tour[j - 1]] + distances[tour[i + 1]][tour[j]]
            if new_distance < old_distance and new_distance < min_distance:
                min_distance = new_distance
                min_index = j
        if min_index != -1:
            tour[i + 1:min_index] = reversed(tour[i + 1:min_index])

    return tour[:-1]
