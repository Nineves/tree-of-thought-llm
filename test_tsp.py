#from src.tot.tasks.tsp import TSPTask
from src.tot.prompts.tsp import *
import tsplib95
import openai

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

response = get_completion(code_prompt, [])
print(response)