#%% Advent of code 2019

#%% Import
import numpy as np
import itertools

#%% Tools
def read_txt(file_name):
    out = []
    v = []
    for x in open(file_name, 'r').readlines():
        n = x.split('\n')[0]
        if len(n) == 0:
            out.append(v)
            v = []
        else:
            v.append(n)
    out.append(v)
    return out

def convert_intcode(input):
    return [int(x) for x in input[0].split(',')]

def get_fuel(mass):
    fuel = np.max([np.floor(mass/3) - 2, 0])
    return fuel

def get_total_fuel(mass):
    if mass > 0:
        return get_fuel(mass) + get_total_fuel(get_fuel(mass))
    else:
        return 0

def traverse_path(path):
    xy_list = []
    xy = (0,0)
    xy_list.append(xy)
    for p in path:
        for i in range(int(p[1:])):
            if p[0] == 'R':
                xy = (xy[0] + 1, xy[1])
                xy_list.append(xy)
            elif p[0] == 'L':
                xy = (xy[0] - 1, xy[1])
                xy_list.append(xy)
            elif p[0] == 'U':
                xy = (xy[0], xy[1] + 1)
                xy_list.append(xy)
            elif p[0] == 'D':
                xy = (xy[0], xy[1] - 1)
                xy_list.append(xy)
    return xy_list

def convert_pass(p):
    return int("".join([str(x) for x in p]))

def convert_int(n):
    return [int(x) for x in str(n)]    

def check_pass(p):
    a = False
    b = False
    for i in range(len(p) - 1):
        if p[i] > p[i+1]:
            a = False
            b = False
            break
        elif int(p[i]) == int(p[i+1]):
            a = True
            if sum([x==p[i] for x in p]) == 2:
                b = True
    return a, b

def next_pass(p):
    for i in range(len(p) - 1, 0, -1):
        if p[i] < p[i-1]:
            p[i] = p[i-1]
            return convert_int(convert_pass(p))
        else:
            return convert_int(convert_pass(p) + 1)

#%%
class intcode_computer:

    def __init__(self, intcode, index=0, inputs=[], outputs=[], debug=False):
        self.intcode = intcode
        self.index = index
        self.inputs = inputs
        self.outputs = outputs
        self.debug = debug
        
    def process_block(self):
        # process the current code block, including interpreting the code, reading the outputs, generating the outputs, and updating the index
    
    def process_code(code):
        full_code = [int(x) for x in str(code)]
        for i in range(5-len(full_code)):
            full_code.insert(0,0)
        full_code = "".join([str(x) for x in full_code])
        opcode = int(full_code[-2:])
        params = [int(x) for x in full_code[:3][::-1]]
        return opcode, params
    
    
        
#%%
def intcode_computer(intcode, intcode_input_list, intcode_output_list=[], idx=0, debug=False):
    
    
    def process_code(code):
        full_code = [int(x) for x in str(code)]
        for i in range(5-len(full_code)):
            full_code.insert(0,0)
        full_code = "".join([str(x) for x in full_code])
        opcode = int(full_code[-2:])
        params = [int(x) for x in full_code[:3][::-1]]
        return opcode, params
    
    
    def get_inputs(idx, opcode, params):
        inputs = []
        if opcode in (1, 2, 7, 8):
            for p, i in zip(params, range(len(params))):
                if (i < 2):
                    if p == 0:
                        inputs.append(intcode[intcode[idx + i + 1]])
                    elif p == 1:
                        inputs.append(intcode[idx + i + 1])
                elif i == 2:
                    inputs.append(intcode[idx + i + 1]) # exception for writing to memory
        elif opcode == 3:
            inputs.append(intcode[idx + 1]) # exception for writing to memory
        elif opcode == 4:
            for p, i in zip(params, range(len(params))):
                if (i < 1):
                    if p == 0:
                        inputs.append(intcode[intcode[idx + i + 1]])
                    elif p == 1:
                        inputs.append(intcode[idx + i + 1])
        elif opcode in (5, 6):
            for p, i in zip(params, range(len(params))):
                if (i < 2):
                    if p == 0:
                        inputs.append(intcode[intcode[idx + i + 1]])
                    elif p == 1:
                        inputs.append(intcode[idx + i + 1])
        return inputs
    
    
    def extend_intcode(d):
        while len(intcode) < (d + 1):
            intcode.append(0)
    
    
    def get_next_idx(idx, opcode):
        if opcode in (1, 2, 7, 8):
            n = 4
        elif opcode in (3, 4):
            n = 2
        elif opcode in (5, 6):
            n = 3
        return int(idx + n)
    
    
    def perform_operations(idx, opcode, inputs):
        
        # wrap up the iteration (might get overwritten later)
        next_idx = get_next_idx(idx, opcode)
        output = None
        output_idx = None
        
        # perform writing operations
        if opcode in (1, 2, 7, 8):
            if opcode == 1:
                output = inputs[0] + inputs[1]
                output_idx = inputs[2]
            elif opcode == 2:
                output = inputs[0] * inputs[1]
                output_idx = inputs[2]
            elif opcode == 7:
                if inputs[0] < inputs[1]:
                    output = 1
                else: 
                    output = 0
                output_idx = inputs[2]
            elif opcode == 8:
                if inputs[0] == inputs[1]:
                    output = 1
                else: 
                    output = 0
                output_idx = inputs[2]
            extend_intcode(output_idx)
            intcode[output_idx] = output
            return next_idx, None
        elif opcode in (5, 6):
            if opcode == 5:
                if inputs[0] != 0:
                    next_idx = inputs[1]
            elif opcode == 6:
                if inputs[0] == 0:
                    next_idx = inputs[1]
            return next_idx, None
        elif opcode == 3:
            output = intcode_input_list.pop(0)
            output_idx = inputs[0]
            extend_intcode(output_idx)
            intcode[output_idx] = output
            return next_idx, None
        elif opcode == 4:
            output = inputs[0]
            return next_idx, output
        else: 
            raise Exception('bad opcode')
        
    
    def iterate_intcode(idx):
        code = intcode[idx]
        opcode, params = process_code(code)
        inputs = get_inputs(idx, opcode, params)
        if opcode == 99:
            return opcode, idx, False, None
        elif (opcode == 3) & (len(intcode_input_list) == 0):
            return opcode, idx, False, None
        next_idx, output = perform_operations(idx, opcode, inputs)
        return opcode, next_idx, True, output
     
    run = True
    intcode_output_list = []
    while run:
        opcode, idx, can_continue, output = iterate_intcode(idx)
        if not can_continue:
            return intcode, idx, intcode_output_list, opcode
        elif opcode == 4:
            intcode_output_list.append(output)


def run_sequential_intcode_computers(n, starting_code, input_array=[]):
    
    # initialize computers
    intcode_list = []
    for i in range(n):
        intcode_list.append(starting_code.copy())
    
    # initilize inputs
    input_list = [[x] for x in input_array]
    input_list[0].append(0)
    idx_list = [0]*n
    
    # run computers
    while True:
        for i in range(n):
            curr_intcode, idx, curr_output, curr_opcode = intcode_computer(intcode_list[i], input_list[i], idx=idx_list[i])
            intcode_list[i] = curr_intcode
            idx_list[i] = idx
            input_list[int(np.mod(i+1,n))] += curr_output
        if curr_opcode == 99:
            break
    return curr_output


#%% Question 1
q1input = read_txt(r'C:\Users\Lenny\Documents\GitHub\AdventOfCode\2019\q1input.txt')[0]
a = 0
b = 0
for x in q1input:
    a += get_fuel(int(x))
    b += get_total_fuel(int(x))
print(a,b)

#%% Question 2
q2input = read_txt(r'C:\Users\Lenny\Documents\GitHub\AdventOfCode\2019\q2input.txt')[0]
q2input = convert_intcode(q2input)


code = q2input.copy()
code[1] = 12
code[2] = 2
code,a,b,c = intcode_computer(code, [])
print(code[0])
for i in range(100):
    for j in range(100):
        code = q2input.copy()
        code[1] = i
        code[2] = j
        code,a,b,c = intcode_computer(code, [])
        if code[0] == 19690720:
            print(100*i + j)
            break

#%% Question 3
q3input = read_txt(r'C:\Users\Lenny\Documents\GitHub\AdventOfCode\2019\q3input.txt')[0]
path1 = q3input[0].split(',')
path2 = q3input[1].split(',')

xy1 = traverse_path(path1)
xy2 = traverse_path(path2)
a = 99999999999999
b = 99999999999999
for s in set(set(xy1) & set(xy2)):
    if s != (0,0):
        a = min(a, abs(s[0]) + abs(s[1]))
        idx1 = xy1.index(s)
        idx2 = xy2.index(s)
        b = min(b, idx1+idx2)
print(a, b)

#%% Question 4
q4input = [int(x) for x in ('273025-767253').split('-')]
min_pass = convert_int(q4input[0])
max_pass = q4input[1]

p = []
p2 = []
t = min_pass.copy()
while True:
    print(t)
    if convert_pass(t) > max_pass:
        break
    elif check_pass(t)[0]:
        p.append(convert_pass(t))
        if check_pass(t)[1]:
            p2.append(convert_pass(t))
    t = next_pass(t)
print(len(p), len(p2))

#%% Question 5
q5input = read_txt(r'C:\Users\Lenny\Documents\GitHub\AdventOfCode\2019\q5input.txt')[0]
q5input = convert_intcode(q5input)
    
o1, o2, o3, o4 = intcode_computer(q5input.copy(), [1])
print(o2)
o1, o2, o3, o4 = intcode_computer(q5input.copy(), [5])
print(o2)

#%% Question 6

#%% Question 7
q7input = read_txt(r'C:\Users\Lenny\Documents\GitHub\AdventOfCode\2019\q7input.txt')[0]
q7input = convert_intcode(q7input)

a = []
for p in list(itertools.permutations(range(0, 5))):    
    a.append(run_sequential_intcode_computers(5, q7input, input_array=p)[0])
print(max(a))

a = []
for p in list(itertools.permutations(range(5, 10))):    
    a.append(run_sequential_intcode_computers(5, q7input, input_array=p)[0])
print(max(a))
        