import math
import tsplib95

def distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calculate_total_length(points, path):
    """Calculate the total length of the path."""
    total_length = 0
    for i in range(len(path)):
        total_length += distance(points[path[i-1]], points[path[i]])
    total_length += distance(points[path[-1]], points[path[0]])
    return total_length

def increament_length(points, current_path, new_vertex):
    """Calculate the increase in length when inserting a new vertex into the current path."""
    min_increament = float('inf')
    for i in range(1, len(current_path)):
        increament = distance(points[current_path[i-1]], points[new_vertex]) + distance(points[new_vertex], points[current_path[i]]) - distance(points[current_path[i-1]], points[current_path[i]])
        if increament < min_increament:
            min_increament = increament
            position = i
    return position, min_increament

def solve_tsp(points):
    """Solve the travelling salesman problem by cheapest insertion."""
    # Start by connecting the first two vertices
    path = [0, 1]
    # Insert the rest vertices
    for _ in range(2, len(points)):
        min_increament = float('inf')
        # Try to insert each vertex into the current path
        for i in set(range(len(points))) - set(path):
            position, increament = increament_length(points, path, i)
            # Choose the vertex that causes the smallest increase in length
            if increament < min_increament:
                min_increament = increament
                best_position = position
                best_vertex = i
        # Insert the chosen vertex
        path.insert(best_position, best_vertex)
    # Make the path a cycle
    path.append(path[0])
    return path

if __name__ == '__main__':
    # Load the data of the problem
    path = r'C:\Users\evely\OneDrive\Documents\GitHub\tree-of-thought-llm-fork\src\tot\data\TSP\ulysses16.tsp'
    problem = tsplib95.load(path)
    points = list(problem.node_coords.values())
    # Solve the problem
    path = solve_tsp(points)
    # Calculate the length of the cycle
    length = calculate_total_length(points, path)
    print(path, length)