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
code = intcode_computer(code)
print(code[0])
for i in range(100):
    for j in range(100):
        code = q2input.copy()
        code[1] = i
        code[2] = j
        code = intcode_computer(code)
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
    
o1, o2 = intcode_computer(q5input.copy(), [1])
print(o2)
o1, o2 = intcode_computer(q5input.copy(), [5])
print(o2)

#%% Question 6

#%% Question 5
q7input = read_txt(r'C:\Users\Lenny\Documents\GitHub\AdventOfCode\2019\q7input.txt')[0]
q7input = convert_intcode(q7input)

def intcode_computer(intcode, intcode_input_list, intcode_output_list=[], intcode_idx_override=None, debug=False):
    
    
    def process_code(code):
        full_code = [int(x) for x in str(code)]
        for i in range(5-len(full_code)):
            full_code.insert(0,0)
        full_code = "".join([str(x) for x in full_code])
        opcode = int(full_code[-2:])
        params = [int(x) for x in full_code[:3][::-1]]
        return opcode, params
    
    
    def get_inputs(idx, opcode, params):
        if opcode in (1, 2, 7, 8):
            inputs = []
            for p, i in zip(params, range(len(params))):
                if (i < 2):
                    if p == 0:
                        inputs.append(intcode[intcode[idx + i + 1]])
                    elif p == 1:
                        inputs.append(intcode[idx + i + 1])
                elif i == 2:
                    inputs.append(intcode[idx + i + 1]) # exception for writing to memory
        elif opcode == 3:
            inputs = []
            inputs.append(intcode[idx + 1]) # exception for writing to memory
        elif opcode == 4:
            inputs = []
            for p, i in zip(params, range(len(params))):
                if (i < 1):
                    if p == 0:
                        inputs.append(intcode[intcode[idx + i + 1]])
                    elif p == 1:
                        inputs.append(intcode[idx + i + 1])
        elif opcode in (5, 6):
            inputs = []
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
        
        # perform any operations
        if opcode == 1:
            output = inputs[0] + inputs[1]
            output_idx = inputs[2]
        elif opcode == 2:
            output = inputs[0] * inputs[1]
            output_idx = inputs[2]
        elif opcode == 3:
            output = intcode_input_list.pop(0)
            output_idx = inputs[0]
        elif opcode == 4:
            output = inputs[0]
        elif opcode == 5:
            if inputs[0] != 0:
                next_idx = inputs[1]
        elif opcode == 6:
            if inputs[0] == 0:
                next_idx = inputs[1]
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
        
        # save any variables
        if opcode in (1, 2, 3, 7, 8):
            extend_intcode(output_idx)
            intcode[output_idx] = output
            
        return next_idx, output, output_idx
        
    
    def iterate_intcode(idx):
        code = intcode[idx]
        opcode, params = process_code(code)
        if opcode == 99:
            return opcode, idx, None, False
        elif (opcode == 3) & (len(intcode_input_list) == 0):
            return opcode, idx, None, True
        inputs = get_inputs(idx, opcode, params)
        next_idx, output, output_idx = perform_operations(idx, opcode, inputs)
        return opcode, next_idx, output, False
     
    if intcode_idx_override is None:
        idx = 0
    else:
        idx = intcode_idx_override
    run = True
    intcode_output_list = []
    while run:
        opcode, idx, output, continue_code = iterate_intcode(idx)
        if opcode == 99:
            return intcode, idx, intcode_output_list, opcode
        elif (opcode == 3) & (continue_code):
            return intcode, idx, intcode_output_list, opcode
        elif opcode == 4:
            intcode_output_list.append(output)


# n = 5
# a = []
# for p in list(itertools.permutations(range(n))):
#     intcode_list = [q7input.copy()]*n
#     input_list = [[x] for x in p]
#     input_list[0].append(0)
#     for i in range(n):
#         o1, idx, o2, o3 = intcode_computer(intcode_list[i], input_list[i])
#         intcode_list[i] = o1
#         input_list[int(np.mod(i+1,n))].append(o2[-1])
#     a.append(o2[-1])
# print(max(a))


n = 5
a = []
# for p in list(itertools.permutations(range(5, 5+n))):

# intcode_list = [q7input.copy()]*n
# p = (4,3,2,1,0); intcode_list = [[3,15,3,16,1002,16,10,16,1,16,15,15,4,15,99,0,0]]*n
# p = (0,1,2,3,4); intcode_list = [[3,23,3,24,1002,24,10,24,1002,23,-1,23,101,5,23,23,1,24,23,23,4,23,99,0,0]]*n
# p = (1,0,4,3,2); intcode_list = [[3,31,3,32,1002,32,10,32,1001,31,-2,31,1007,31,0,33,1002,33,7,33,1,33,31,31,1,32,31,31,4,31,99,0,0,0]]*n
p = (9,8,7,6,5); intcode_list = [[3,26,1001,26,-4,26,3,27,1002,27,2,27,1,27,26,27,4,27,1001,28,-1,28,1005,28,6,99,0,0,5]]*n
input_list = [[x] for x in p]
input_list[0].append(0)
idx_list = [0]*n
end_count = [0]*n
k = 0
while True:
    for i in range(n):
        o1, idx, o2, o3 = intcode_computer(intcode_list[i], input_list[i], intcode_idx_override=idx_list[i])
        intcode_list[i] = o1
        idx_list[i] = idx
        input_list[int(np.mod(i+1,n))] += o2
        if o3 == 99:
            end_count[i] += 1
    k += 1
    if sum(end_count) >1000:
        break
a.append(o2)

print(max(a))


        