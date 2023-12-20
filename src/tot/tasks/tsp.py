from tot.tasks.base import Task, DATA_PATH
from tot.prompts.tsp import *
import tsplib95
import os
import re


class TSPTask(Task):
    """
    Input: a list of nodes with coordinates
    Output: a shortest hamiltonian cycle
    Cost: the total length of the cycle
    """

    def __init__(self, file = 'ulysses16.tsp'):
        super().__init__()

        path = os.path.join(DATA_PATH, 'TSP', file)
        self.problem = tsplib95.load(path)
        self.N = len(self.problem.node_coords)
        self.points = [i for i in range(self.N)]
        self.coords = list(self.problem.node_coords.values())
        self.value_cache = {}
    
    @staticmethod
    def insert(trace: list, interval: list, point: int):
        """
        Insert a point into a trace

        """
        # Finding the index of the interval to insert the point
        idx = 0
        while idx < len(trace) - 1 and not (trace[idx] == interval[0] and trace[idx + 1] == interval[1]):
            idx += 1

        # Inserting the point
        trace.insert(idx + 1, point)
        return trace
    
    def calculate_length(self, trace: list):
        """
        Calculate the length of a trace
        """
        length = 0
        for i in range(len(trace) - 1):
            length += self.problem.get_weight(trace[i], trace[i + 1])
        length += self.problem.get_weight(trace[-1], trace[0])

        return length
        
    def test_output(self, output: str):
        """
        Output format: <trace> 0, 1, 2, 3, 4 </trace>\n <point> 5 </point>\n <interval> 2, 3 </interval>
        """
        # Regular expression to match the content between tags
        pattern = r'<(\w+)>\s*([\d,\s]+)\s*</\1>'

        # Finding all matches
        matches = re.findall(pattern, output)

        # Result dictionary
        result = {'trace': [], 'point': [], 'interval': []}

        # Processing each match
        for tag, values in matches:
            # Splitting the values and converting them to integers
            int_values = [int(val.strip()) for val in values.split(',')]

            # Adding the list to the corresponding tag in the result
            result[tag].append(int_values)
        
        new_trace = self.insert(result['trace'][0], result['interval'][0], result['point'][0][0])
        if sorted(new_trace) != sorted(result['trace'][0]):
            return {'r': -1}
        else:
            return {'r': self.calculate_length(new_trace)}
    
    @staticmethod
    def initial_prompt_wrap(points):
        return initial_prompt.format(list_of_points = points)







        

    
        
