import numpy as np
import heapq

def calculate_distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def greedy_tour(points):
    remaining_points = points.copy()
    start = remaining_points.pop(0)
    tour = [start]
    while remaining_points:
        next_point = min(remaining_points, key=lambda x: calculate_distance(tour[-1], x))
        remaining_points.remove(next_point)
        tour.append(next_point)
    return tour

def calculate_tour_length(tour):
    return sum(calculate_distance(tour[i-1], tour[i]) for i in range(len(tour)))

def two_opt_swap(tour, i, j):
    new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
    return new_tour

def solve_tsp(points):
    tour = greedy_tour(points)
    improvement = True
    while improvement:
        improvement = False
        for i in range(1, len(points) - 1):
            for j in range(i+1, len(points)):
                new_tour = two_opt_swap(tour, i, j)
                if calculate_tour_length(new_tour) < calculate_tour_length(tour):
                    tour = new_tour
                    improvement = True
    return [points.index(point) for point in tour]