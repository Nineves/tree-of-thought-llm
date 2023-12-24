import numpy as np
import random
from scipy.spatial import distance_matrix
from multiprocessing import Pool

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

def calculate_swap_impact(tour, i, j):
    old_distance = calculate_distance(tour[i-1], tour[i]) + calculate_distance(tour[j-1], tour[j])
    new_distance = calculate_distance(tour[i-1], tour[j]) + calculate_distance(tour[i], tour[j-1])
    return old_distance - new_distance

def solve_tsp(points):
    best_tour = []
    best_distance = float('inf')

    for _ in range(10):  # run the algorithm 10 times with different starting points
        tour = nearest_neighbor_algorithm(points)
        limit = 0
        while limit < 100:
            is_impact = False
            swap_impacts = [(i, j, calculate_swap_impact(tour, i, j)) for i in range(len(tour)-1) for j in range(i+1, len(tour))]
            swap_impacts.sort(key=lambda x: x[2], reverse=True)  # sort the swaps based on their impact
            for i, j, impact in swap_impacts:
                if impact > 0:  # if the swap reduces the tour length
                    tour = two_opt_swap(tour, i, j)
                    is_impact = True
                    break
            if not is_impact:  # if no swap reduces the tour length
                break
            limit += 1

        distance = calculate_tour_length(tour)
        if distance < best_distance:
            best_tour = tour
            best_distance = distance

    return [points.index(point) for point in best_tour]