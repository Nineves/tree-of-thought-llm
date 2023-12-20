insert_prompt = '''

You are going to solve a travelling salesman problem. You are given a list of points with coordinates below:
{list_of_points}
You are given the current trace:
{trace}
Now select a point from the list and select two neighboring points to insert it, so that the total length of the complete trace is likely to be minimized. For example: <point> 5 </point> <interval> 2, 3 </interval>. The selected point should not be in current trace.
Output:

'''

insert_prompt_reasoning = '''
You are going to solve a travelling salesman problem. You are given a list of points with coordinates below:
{list_of_points}
You are given the current trace:
{trace}
What are some effective strategies to select a point and two neighboring points to insert it, so that the total length of the complete trace is likely to be minimized? Please list out {N} most important strategies starting with <steps> and end with </steps>.


'''

initial_prompt = '''
You are going to solve a travelling salesman problem. You are given a list of points with coordinates below:
{list_of_points}
Now select two points to form the first edge of the target trace. Here are some useful hints for the selection:
{steps}
The format of the output should be <interval> two points </interval>. For example: <interval> 1, 3 </interval>
Output:

'''

initial_prompt_code = '''
You are going to solve a travelling salesman problem. You are given a list of points with coordinates below:
{list_of_points}
Now generate python code for the first two points selection. Here are some useful hints for the selection:
{steps}
The output should begin with <code> and end with </code>
Output:
'''

initial_prompt_code_no_hint = '''
You are going to solve a travelling salesman problem. You are given a list of points with coordinates below:
{list_of_points}
Now generate python code for the first two points selection. The output should begin with <code> and end with </code>
Output:
'''

initial_prompt_reasoning = '''
You are going to solve a travelling salesman problem. You are given a list of points with coordinates below:
{list_of_points}
What are some good traits of the first selected two points in order to minimize the complete trace? Please list out {N} most important steps starting with <steps> and end with </steps>.
'''
