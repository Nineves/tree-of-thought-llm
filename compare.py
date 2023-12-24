from code_results_2 import code_1, code_2, code_3, code_4
import tsplib95
import math
import gurobipy as gp

def calculate_cycle_length(points, coords):
    """Calculate the length of the cycle passing through all the points."""
    length = 0
    def distance(point1, point2):
        """Calculate Euclidean distance between two points."""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    for i in range(len(points)):
        length += distance(coords[points[i-1]], coords[points[i]])
    length += distance(coords[points[-1]], coords[points[0]])
    return length

# read tsp data
path = r'C:\Users\evely\OneDrive\Documents\GitHub\tree-of-thought-llm-fork\src\tot\data\TSP\tsp225.tsp'
problem = tsplib95.load(path)
points = list(problem.node_coords.values())



# run code 1
path_1 = code_1.solve_tsp(points)
cost_1 = calculate_cycle_length(path_1, points)

# run code 2
path_2 = code_2.solve_tsp(points)
cost_2 = calculate_cycle_length(path_2, points)

# run code 3
path_3 = code_3.solve_tsp(points)
cost_3 = calculate_cycle_length(path_3, points)

# run code 4
#path_4 = code_4.solve_tsp(points)
#cost_4 = calculate_cycle_length(path_4, points)

# compare results
print('Code 1:', path_1, cost_1)
print('Code 2:', path_2, cost_2)
print('Code 3:', path_3, cost_3)
#print('Code 4:', path_4, cost_4)
