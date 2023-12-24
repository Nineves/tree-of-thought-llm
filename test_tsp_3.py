#from src.tot.tasks.tsp import TSPTask
from src.tot.prompts.tsp import *
import tsplib95
import openai
import numpy as np
import itertools

openai.api_key = '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'
def get_completion(prompt, message_history, model="gpt-4"):
    messages = message_history + [{"role": "user", "content": prompt}]
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.9
        )
        return response.choices[0].message["content"]
    except openai.error.OpenAIError as e:
        print("OpenAI API error:", e)
        return None
    

path = r'C:\Users\evely\OneDrive\Documents\GitHub\tree-of-thought-llm-fork\src\tot\data\TSP\ulysses16.tsp'
tsp_problem = tsplib95.load(path)
tsp_points = list(tsp_problem.node_coords.values())
points = [i for i in range(len(tsp_points))]
prompt =  ""
for node in points:
        if node == 0 or node == 2:
             continue
        prompt += f"{node}: {tuple(tsp_points[node])}, "
points = points[:-2]

steps = '''
Location: In some cases, the first two selected points can be near the center of all points. This can be beneficial as it could mean less distance is traveled on the way back to the starting point at the end of the tour.

Proximity: The first two selected points should ideally be the closest to each other. This is because the shortest distance between two points is a straight line. Therefore, choosing two points that are close together as the starting point will help to reduce the overall tour length.
 
Outliers: If there are any outlier points that are far away from the other points, it might be a good idea to choose one of these as one of the first two points. This is because if they are not visited early on, they could potentially increase the overall tour length significantly later on.
'''

insertion_steps = '''

1. Nearest Neighbor: This strategy focuses on adding the point that is nearest to the current point in the trace. It is a greedy algorithm and very easy to implement.

2. Cheapest Insertion: Add points to the trace which have the smallest increase in total length. This could be seen as a compromise between inserting the nearest point and trying to keep the tour as short as possible.

3. Furthest Insertion: Similar to cheapest insertion, but instead of always adding the point with the least increase in total length, the point with the largest decrease in total length when removed from the remaining points is added to the trace.

4. Random Insertion: In this strategy, points are randomly chosen and added to the trace. This strategy is often used in conjunction with other strategies 
to add some randomness and avoid getting stuck in local minima.
'''
#complete_p  =initial_prompt_code_no_hint.format(list_of_points = prompt, steps = steps)
#complete_p  =initial_prompt_code.format(list_of_points = prompt, steps = steps)
complete_p  =insert_prompt_reasoning.format(list_of_points = prompt, N = 4, trace= "0, 2")
print(complete_p)

code_prompt = '''
The strategy that will be used for finding the optimal TSP path is:
Cheapest Insertion: Similar to cheapest insertion, but instead of always adding the point with the least increase in total length, the point with the largest decrease in total length when removed from the remaining points is added to the trace.
The skeleton code is given below:
<code>
def solve_tsp(points):
:points: a list of points with coordinates, e.g.[(0, 0), (0, 1), (1, 0), (1, 1), (2, 1)]
:return: a shortest hamiltonian cycle, e.g. [0, 1, 2, 3, 4]

# your code here

return path

</code>
Now generate python code for deriving the shortest valid path. You need to write the code based on the strategy you have chosen and the skeleton code given above.
Output:

'''

improve_code_prompt ='''
You are going to solve a travelling salesman problem. You are given a list of points and you need to find the shortest hamiltonian cycle. 
Below is the code for the initial solution generated:
<code>
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

</code>
First, you need to describe the solution from generated code within five steps, be succinct and precise. The output should begin with <steps> and end with </steps>. For example: <steps> First, calculate the Euclidean distance between two points </steps>
Output:
'''

improve_code_prompt_2 ='''
You are going to solve a travelling salesman problem. You are given a list of points and you need to find the shortest hamiltonian cycle. 
Below is the code for the initial solution generated:
<code>
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

</code>
Below is the description of the solution from generated code:
<steps> 
1. The function first calculates the Euclidean distance between every two points and saves it in a distance matrix.        
2. Initially, the travelling salesman path (tsp_path) contains only the first point and all other points are in the list of available points.
3. In each iteration of the while loop, the function finds the point in the set of available points that, when added to the current path, results in the smallest increase of the total path length.
4. The identified point is added to the path in the position following the point after which it is to be inserted to achieve the minimal distance.
5. The process continues until there are no more points left in the set of available points. The function then returns the 
constructed path.
</steps>
Now you need to inspect the solution carefully and think creatively which parts of the code can be improved. 
You need to describe the improved solution from generated code in steps, be succinct and precise. 
The output should begin with <steps> and end with </steps>. For example: <steps> First, calculate the Euclidean distance between two points </steps>
'''

improve_code_prompt_3 = '''
You are going to solve a travelling salesman problem. You are given a list of points and you need to find the shortest hamiltonian cycle. 
Below is the code for the initial solution generated:
<code>
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

</code>
Below is the suggested improvements from generated code:
<steps>
1. The function can be improved by precomputing the distance matrix and storing it in memory. This way, the distance between two points needs to be calculated only once and can be looked up from the matrix during subsequent iterations. This will 
significantly reduce computational time if the number of points is large.
2. Instead of going through the entire list of available points each time when looking for the next point to add to the path, you can maintain a priority queue that sorts the points by the increase in total path length if they are added to the path. This will reduce the time complexity of each iteration from O(n) to O(logn), where n is the number of points.
3. The current implementation updates min_distance and associated variables inside the nested loop. This can result in multiple unnecessary updates if more than one point yields the same minimal increase in path length. To avoid this, you can calculate the increases for all points first and then find the minimum value and the associated point and insertion position. 
This will also make the code more readable.
4. The function can further be optimized by applying a heuristic algorithm like the 2-opt technique, which iteratively removes two edges from the path and reconnects the path in a different way if it results in a shorter path. Such local optimization can significantly improve the solution, especially for larger inputs.
5. Instead of manipulating the list of available points by removing the selected point in each iteration, you can maintain 
a boolean array to indicate whether each point has been added to the path. This way, you can skip the points that have been added when scanning for the next point to add, without changing the structure of the array.
</steps>
The skeleton code is given below:
<code>
def solve_tsp(points):
:points: a list of points with coordinates, e.g.[(0, 0), (0, 1), (1, 0), (1, 1), (2, 1)]
:return: a shortest hamiltonian cycle, e.g. [0, 1, 2, 3, 4]

# your code here

return path

</code>
Now generate python code for deriving the shortest valid path. You need to write the code based on the strategy you have chosen and the skeleton code given above. The output should begin with <code> and end with </code>
Output:


'''

improve_code_prompt_4 = '''
You are going to solve a travelling salesman problem. You are given a list of points and you need to find the shortest hamiltonian cycle. 
Below is the code for the last solution generated:
<code>
import numpy as np
import heapq

def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def solve_tsp(points):
    n = len(points)

    # Precompute distance matrix
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance_matrix[i, j] = calculate_distance(points[i], points[j])

    # Create a boolean array to track points in the path
    in_path = [False]*n
    # Start from the first point
    in_path[0] = True
    tsp_path = [0]

    # Initialize priority queue
    pq = []

    for point in range(1, n):
        increment = distance_matrix[0][point]
        heapq.heappush(pq, (increment, point, 0))

    while pq:
        increment, point, i = heapq.heappop(pq)
        if in_path[point]: continue

        in_path[point] = True
        tsp_path.insert(i+1, point)

        for next_point in range(0, n):
            if not in_path[next_point]:
                if i+1 == len(tsp_path) - 1:
                    new_increment = distance_matrix[tsp_path[i+1]][next_point] + distance_matrix[next_point][tsp_path[0]] - distance_matrix[tsp_path[i+1]][tsp_path[0]]
                else:
                    new_increment = distance_matrix[tsp_path[i+1]][next_point] + distance_matrix[next_point][tsp_path[i+2]] - distance_matrix[tsp_path[i+1]][tsp_path[i+2]]
                heapq.heappush(pq, (new_increment, next_point, i+1))
    return tsp_path

</code>
First, you need to understand the logic of the generated code from the above code. 
Now you need to inspect the solution carefully and think creatively which parts of the code can be improved to minimize the path length. For example, you may add new algorithms or change the existing algorithms.
You need to describe the improved solution from generated code in steps, be succinct and precise. 
The output should begin with <steps> and end with </steps>. For example: <steps> First, calculate the Euclidean distance between two points </steps>
'''

improve_code_prompt_5 = '''
You are going to solve a travelling salesman problem. You are given a list of points and you need to find the shortest hamiltonian cycle. 
Below is the code for the last solution generated:
<code>
import numpy as np
import heapq

def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def solve_tsp(points):
    n = len(points)

    # Precompute distance matrix
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance_matrix[i, j] = calculate_distance(points[i], points[j])

    # Create a boolean array to track points in the path
    in_path = [False]*n
    # Start from the first point
    in_path[0] = True
    tsp_path = [0]

    # Initialize priority queue
    pq = []

    for point in range(1, n):
        increment = distance_matrix[0][point]
        heapq.heappush(pq, (increment, point, 0))

    while pq:
        increment, point, i = heapq.heappop(pq)
        if in_path[point]: continue

        in_path[point] = True
        tsp_path.insert(i+1, point)

        for next_point in range(0, n):
            if not in_path[next_point]:
                if i+1 == len(tsp_path) - 1:
                    new_increment = distance_matrix[tsp_path[i+1]][next_point] + distance_matrix[next_point][tsp_path[0]] - distance_matrix[tsp_path[i+1]][tsp_path[0]]
                else:
                    new_increment = distance_matrix[tsp_path[i+1]][next_point] + distance_matrix[next_point][tsp_path[i+2]] - distance_matrix[tsp_path[i+1]][tsp_path[i+2]]
                heapq.heappush(pq, (new_increment, next_point, i+1))
    return tsp_path

</code>
Below is the description of the possible improvements from generated code:
<steps>
1. Instead of using a Boolean array to track points in the path, you can use a set. This would optimize look-up operations 
to O(1) instead of O(n) complexity.

2. The distance matrix computation part of the code has a time complexity of O(n^2). To reduce this, you can try using a more efficient algorithm like Floyd-Warshall or Johnson's algorithm which calculates all pairs shortest path in O(n^2 logn) and O(n^2) time respectively.

3. The solution above uses a greedy approach for the choice of next point on the TSP path. It may work fine for smaller inputs, but for larger inputs, it would often fall into a local optimum solution. You can use an optimization method that considers global optima like Simulated Annealing or Genetic Algorithms instead of the greedy approach. This would improve the quality of the solution.

4. The priority queue used in the solution adds an additional time complexity of O(n logn). Instead of a priority queue, use a Min-heap data structure to improve the time complexity.

5. The insertion of points in the middle of the TSP path would be costly in terms of time complexity. To optimize it, you can use a linked list for tsp_path, which allows O(1) time complexity for insertions in the middle.

6. Finally, the while loop in the code has a worst-case time complexity of O(n^2 logn). You can use a more efficient loop condition or use a for loop instead to reduce the time complexity.
</steps>
The skeleton code is given below:
<code>
def solve_tsp(points):
:points: a list of points with coordinates, e.g.[(0, 0), (0, 1), (1, 0), (1, 1), (2, 1)]
:return: a shortest hamiltonian cycle, e.g. [0, 1, 2, 3, 4]

# your code here

return path

</code>
Now generate python code for deriving the shortest valid path. You need to write the code based on the above information. The output should begin with <code> and end with </code>


'''


improve_code_prompt_6 = '''
You are going to solve a travelling salesman problem. You are given a list of points and you need to find the shortest hamiltonian cycle. 
Below is the code for the last solution generated:
<code>
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

</code>
First, you need to understand the logic of the generated code from the above code. 
Now you need to inspect the solution carefully and think creatively which parts of the code can be improved to minimize the path length. For example, you may add new algorithms or change the existing algorithms.
You need to describe the improved solution from generated code in steps, be succinct and precise. 
The output should begin with <steps> and end with </steps>. For example: <steps> First, calculate the Euclidean distance between two points </steps>
'''

improve_code_prompt_7 = '''
You are going to solve a travelling salesman problem. You are given a list of points and you need to find the shortest hamiltonian cycle. 
Below is the code for the last solution generated:
<code>
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

</code>
Below are the possible improvements for the generated code:
<steps>
1. Use a more sophisticated approach for generating the initial tour: The current solution starts a tour from the first point. This is a simple, but not always optimal strategy. Instead, you might want to use a more sophisticated approach like "Nearest Neighbor" which starts at a random point, but then chooses the nearest point as the next one.

2. Use 2-opt or 3-opt local search for improving the tour: After you have a tour, the solution just returns it as is. However, you can often significantly shorten the tour by iteratively applying "2-opt" or "3-opt" local search, which removes two or three edges from the tour and reconnects it in a different way, if it results in a shorter tour.

3. Parallelization: The current solution doesn't make use of parallelization although many parts of the problem are highly 
parallelizable. For example, you can calculate the distance matrix in parallel.

4. Use a more efficient data structure: Heap is used to store and retrieve the distances. Priority queue could be a better 
alternative which may reduce the time complexity.

5. Heuristics: There are efficient heuristic algorithms like Simulated Annealing, Genetic Algorithms, Ant Colony Optimization, etc for TSP problems. Implementing these might result in a better solution.

6. Pruning: In the current solution, all distances are pushed into heap regardless of whether they will ever be used or not. A better approach would be to only push into the heap the distances that could potentially lead to a shorter path.       

7. Use the triangle inequality: If the distance function obeys the triangle inequality, this property can be used to eliminate many possible routes and hence speed up the search.
</steps>
The skeleton code is given below:
<code>
def solve_tsp(points):
:points: a list of points with coordinates, e.g.[(0, 0), (0, 1), (1, 0), (1, 1), (2, 1)]
:return: a shortest hamiltonian cycle, e.g. [0, 1, 2, 3, 4]

# your code here

return path

</code>
Now generate python code for deriving the shortest valid path. You need to write the code based on the above information and suggestions. The output should begin with <code> and end with </code>

'''


response = get_completion(improve_code_prompt_7, [])
print(response)