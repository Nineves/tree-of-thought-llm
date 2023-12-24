import numpy as np

def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def solve_tsp(points):
    n = len(points)

    # Initialize distance matrix
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance_matrix[i, j] = calculate_distance(points[i], points[j])

    # Start from the first point
    tsp_path = [0]
    # Available points is a list of all points excluding the starting point
    available_points = list(range(1, n))

    while available_points:
        # Initialize min_distance with a very high value
        min_distance = float('inf')

        for point in available_points:
            for i in range(len(tsp_path)):
                # Calculate the increase in length if the current point is inserted after point i in tsp_path
                if i == len(tsp_path) - 1:
                    increment = distance_matrix[tsp_path[i]][point] + distance_matrix[point][tsp_path[0]] - distance_matrix[tsp_path[i]][tsp_path[0]]
                else:
                    increment = distance_matrix[tsp_path[i]][point] + distance_matrix[point][tsp_path[i + 1]] - distance_matrix[tsp_path[i]][tsp_path[i + 1]]

                # Update min_distance and associated variables if a smaller increase is found
                if increment < min_distance:
                    min_distance = increment
                    insert_after = i
                    min_point = point

        # Insert the point in tsp_path at the correct position
        tsp_path.insert(insert_after + 1, min_point)
        # Remove the point from available_points
        available_points.remove(min_point)

    return tsp_path