import numpy as np
from scipy.spatial import distance_matrix
import random
import itertools

def calculate_distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def nearest_neighbor_algorithm(points):
    remaining_points = points.copy()
    start = remaining_points.pop(random.randint(0, len(remaining_points)-1))
    tour = [start]
    while remaining_points:
        next_point = min(remaining_points, key=lambda x: calculate_distance(tour[-1], x))
        remaining_points.remove(next_point)
        tour.append(next_point)
    return tour

def two_opt_swap(tour, i, j):
    new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
    return new_tour

def three_opt_swap(tour, i, j, k):
    possibilities = [tour[:i] + tour[i:j][::-1] + tour[j:k][::-1] + tour[k:],
                     tour[:i] + tour[j:k]       + tour[i:j][::-1] + tour[k:],
                     tour[:i] + tour[j:k][::-1] + tour[i:j]       + tour[k:]]
    return min(possibilities, key=calculate_tour_length)

def calculate_tour_length(tour):
    return sum(calculate_distance(tour[i-1], tour[i]) for i in range(len(tour)))

def solve_tsp(points):
    tour = nearest_neighbor_algorithm(points)
    improvement = True
    while improvement:
        improvement = False
        for i in range(len(points)-1):
            for j in range(i+1, len(points)):
                new_tour = two_opt_swap(tour, i, j)
                if calculate_tour_length(new_tour) < calculate_tour_length(tour):
                    tour = new_tour
                    improvement = True
                else:
                    for k in range(j+1, len(points)):
                        new_tour = three_opt_swap(tour, i, j, k)
                        if calculate_tour_length(new_tour) < calculate_tour_length(tour):
                            tour = new_tour
                            improvement = True
    return [points.index(point) for point in tour]