#from src.tot.tasks.tsp import TSPTask
from src.tot.prompts.tsp import *
import tsplib95
import openai
import numpy as np
import itertools
# best: 1 ->  3 ->  2 ->  4 ->  8 ->  15 ->  5 ->  11 ->  9 ->  10 ->  7 ->  6 ->  14 ->  13 ->  12 ->  16

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
Cheapest Insertion: Add points to the trace which have the smallest increase in total length. This could be seen as a compromise between inserting the nearest point and trying to keep the tour as short as possible.
For each pair of vertices, there is at most one edge between them. Find a path that visits every vertex in the graph exactly once with shortest total length. 
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

</code>
First, you need to describe the solution from generated code within five steps, be succinct and precise. The output should begin with <steps> and end with </steps>. For example: <steps> First, calculate the Euclidean distance between two points </steps>
Output:
'''

improve_code_prompt_2 ='''
You are going to solve a travelling salesman problem. You are given a list of points and you need to find the shortest hamiltonian cycle. 
Below is the code for the initial solution generated:
<code>
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

</code>
Below is the description of the solution from generated code:
<steps> 
1. The function first calculates the Euclidean distance between every pair of points and stores these distances in a 2D list.
2. It then initializes a random tour that visits each point once and then returns to the starting point.
3. The function then starts to refine the tour. For each point in the tour (except the last two points), it checks if swapping this point with a following 
point would result in a shorter tour.
4. If a shorter tour is found by swapping, the function makes the swap.
5. The process continues until no more profitable swaps can be found, and then it returns the tour.
</steps>
Now you need to inspect the solution carefully and think creatively which parts of the code can be improved. 
You need to describe the improved solution from generated code in steps, be succinct and precise. 
The output should begin with <steps> and end with </steps>. For example: <steps> First, calculate the Euclidean distance between two points </steps>
'''

improve_code_prompt_3 = '''
You are going to solve a travelling salesman problem. You are given a list of points and you need to find the shortest hamiltonian cycle. 
Below is the text description of the solution from generated code:
<steps>
1. Start by calculating the Euclidean distance between all pairs of points and store these distances in a 2D list.
2. Instead of initializing a random tour, use a greedy heuristic for the initial tour. Start from a random point, then at each step visit the closest unvisited point. After all points are visited, return to the starting point. This usually gives a much better initial tour than a random one.
3. Implement a 2-opt swapping to refine the tour. For each pair of edges in the tour, check if swapping them results in a shorter tour. If it does, perform the swap. Repeat this process until no more profitable swaps can be found.
4. To improve the efficiency of the 2-opt swapping, instead of checking all pairs of edges, only check pairs of edges that are likely to result in an improvement. For example, only check pairs of edges that are close to each other in the Euclidean space.
5. Another potential improvement is to use a more advanced local search algorithm for tour refinement, such as 3-opt or simulated annealing. These algorithms have a higher computational complexity, but they usually find better solutions than the simple 2-opt swapping.
6. Lastly, use a priority queue to store the potential swaps, with the priority being the amount of tour length reduction. This way, the function always tries the most promising swaps first, which can greatly speed up the tour refinement process.
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
def two_opt_swap(tour, i, j):
    new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
    return new_tour

def solve_tsp(points):
    tour = greedy_tour(points)
    improvement = True
    while improvement:
            for j in range(i+1, len(points)):
                new_tour = two_opt_swap(tour, i, j)
                    tour = new_tour
                    improvement = True
    return [points.index(point) for point in tour]

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
def two_opt_swap(tour, i, j):
    new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
    return new_tour

def solve_tsp(points):
    tour = greedy_tour(points)
    improvement = True
    while improvement:
            for j in range(i+1, len(points)):
                new_tour = two_opt_swap(tour, i, j)
                    tour = new_tour
                    improvement = True
    return [points.index(point) for point in tour]

</code>
Below is the description of the possible improvements from generated code:
<steps>
1. Critique the existing greedy approach: The current approach uses a greedy algorithm to generate the initial tour. This can certainly be improved as greedy algorithms do not necessarily provide the optimal solution. Instead, we can use a more robust initial tour generation algorithm like the Nearest Neighbor Algorithm.

2. Implementing Nearest Neighbor Algorithm: Start at any city, at each step visit the nearest city that hasn't been visited yet. This way, we can get a better initial tour which might lead to a shorter final tour.

3. Improving the 2-opt Swap: The current 2-opt Swap approach can be made more efficient. Rather than swapping for every pair of edges, we can stop if there is no further improvement in the tour length. We can achieve this by adding a condition to stop the while loop if the tour length before and after the swap is the same.

4. Adding 3-opt Swap: The 2-opt swap strategy can be improved by implementing the 3-opt swap method. This swap involves detaching three edges and reattaching them in a different way. In certain scenarios, this will give a shorter tour.

5. Implementing Simulated Annealing: Adding a simulated annealing strategy for optimization can yield better results. This makes the solution less likely to be stuck in local minimums.

6. Parallelizing the algorithm: Finally, try to parallelize the steps of the algorithm, if possible. This can highly reduce the computational time, especially on larger inputs.

7. Fine Tune and Repeat: Measure the performance of the improved algorithm and tweak the parameters for better results. Continue this process until no significant improvement is observed.
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

</code>
Below is the description of the possible improvements from generated code:
<steps>
1. The initial code uses a nearest neighbor algorithm to generate the initial tour. This algorithm simply picks a random point and then always moves to the nearest unvisited point. The problem with this approach is that it may lead to a locally optimal solution, but often misses the globally optimal solution. An alternate approach could be to use a more complex tour initialization algorithm, such as the Simulated Annealing or Genetic Algorithm. These algorithms may be more time-consuming but are more likely to result in a better initial tour.

2. In the current code, the two-opt and three-opt swaps are performed in a loop until no further improvements can be made. However, these swaps are done in a sequential manner, which may not be the most efficient way. A possible improvement could be to perform these swaps in parallel using multi-threading or 
multiprocessing, which can potentially speed up this optimization process.

3. The existing code simply checks the two-opt and three-opt swaps in the order in which they are defined. An alternate approach could be to sort the swaps based on their potential impact on the tour length. In the case of the two-opt swap, we can calculate the difference in distance between the two edges being swapped and the two edges that would replace them if the swap was made. We can then sort all possible two-opt swaps based on this difference, and check 
the swaps in this order. This would ensure that the most promising swaps are checked first.

4. The code currently calculates the tour length each time it checks a swap. This is unnecessary and can be optimized. Instead of calculating the full tour length, we can simply calculate the difference in distance caused by the swap. This would be equal to the total distance of the new edges minus the total 
distance of the old edges.

5. Once the two-opt and three-opt optimizations are finished, we can further optimize the solution using other optimization methods. For example, the Lin-Kernighan algorithm is a more advanced heuristic that can be used in addition to or instead of the two-opt and three-opt swaps.

6. The selection of the initial point in the nearest neighbor algorithm is random. However, this selection could influence the quality of the final solution. One possible improvement could be to run the algorithm multiple times with different starting points and then select the best result.

7. The current code does not incorporate any form of machine learning in its solution. It might be possible to train a machine learning model like a neural network on a set of known TSP instances and then use the model to predict the optimal tour for new instances.
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


response = get_completion(improve_code_prompt_7, [])
print(response)