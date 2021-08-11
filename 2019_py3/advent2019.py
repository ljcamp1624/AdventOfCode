#%% Advent of code 2019
import numpy as np
import scipy.sparse as sp
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import time
from matplotlib.animation import FuncAnimation

#%% Question 1
def get_fuel(mass):
    fuel = np.max([np.floor(mass/3) - 2, 0]);
    return fuel;
def get_total_fuel(mass):
    fuel = get_fuel(mass);
    total_fuel = fuel;
    while fuel > 0:
        fuel = get_fuel(fuel);
        total_fuel = total_fuel + fuel;
    return total_fuel;

mass = pd.read_csv(r'C:\Users\Leonard\Documents\AdventOfCode2019/q1.csv', header=None);
sum_fuel1 = 0;
sum_fuel2 = 0;
for m in mass[0]:
    sum_fuel1 = sum_fuel1 + get_fuel(m);
    sum_fuel2 = sum_fuel2 + get_total_fuel(m);
print(sum_fuel1);
print(sum_fuel2);

#%% Question 2
def run_intcode(x):
    
    y = x.copy();
    i = 0;
    
    while True:
        code = y[i];
        
        if code == 1 or code == 2:
            
            p1 = y[i+1];
            p2 = y[i+2];
            p3 = y[i+3];
            i = i + 4;
            
            a = y[p1];
            b = y[p2];
            
            if code == 1:
                y[p3] = a+b;
            elif code == 2:
                y[p3] = a*b;
            
        elif code == 99:
            break;
            
        else:
            print('error');
    
    return y[0];

# q2 input
x = np.array([1,0,0,3,1,1,2,3,1,3,4,3,1,5,0,3,2,13,1,19,1,6,19,23,2,6,23,27,1,5,27,31,2,31,9,35,1,35,5,39,1,39,5,43,1,43,10,47,2,6,47,51,1,51,5,55,2,55,6,59,1,5,59,63,2,63,6,67,1,5,67,71,1,71,6,75,2,75,10,79,1,79,5,83,2,83,6,87,1,87,5,91,2,9,91,95,1,95,6,99,2,9,99,103,2,9,103,107,1,5,107,111,1,111,5,115,1,115,13,119,1,13,119,123,2,6,123,127,1,5,127,131,1,9,131,135,1,135,9,139,2,139,6,143,1,143,5,147,2,147,6,151,1,5,151,155,2,6,155,159,1,159,2,163,1,9,163,0,99,2,0,14,0]);
y = x.copy();
y[1] = 12;
y[2] = 2;
print(run_intcode(y));
for n in range(100):
    for m in range(100):
        y = x.copy();
        y[1] = n;
        y[2] = m;
        if run_intcode(y) == 19690720:
            print(n,m,100*n+m);
            
#%% Question 3
def interpret_path(p):
    d = p[0];
    n = int(p[1:]);
    array = np.arange(1, n+1)[:, np.newaxis];
    if d == 'U':
        vec = np.array([0, 1]);
        array = np.append(np.zeros([n, 1]), array, axis=1);
    elif d == 'R':
        vec = np.array([1, 0]);
        array = np.append(array, np.zeros([n, 1]), axis=1);
    elif d == 'D':
        vec = np.array([0, -1]);
        array = np.append(np.zeros([n, 1]), -array, axis=1);
    elif d == 'L':
        vec = np.array([-1, 0]);
        array = np.append(-array, np.zeros([n, 1]), axis=1);
    step = n*vec;
    return step, array;
    
def get_coords(path):
    xy_curr = np.array([0, 0])[np.newaxis, :];
    xy = np.array([0, 0])[np.newaxis, :];
    for p in path:
        step, array = interpret_path(p);            
        xy_new = xy_curr + array;
        xy = np.append(xy, xy_new, axis=0);
        xy_curr = xy_new[-1][np.newaxis, :];
    return xy;
        
p1 = list(pd.read_csv(r'C:\Users\Lenny\Documents\GitHub\AdventOfCode\2019_py3\q3p1.csv', header=None).iloc[0, :]);
p2 = list(pd.read_csv(r'C:\Users\Lenny\Documents\GitHub\AdventOfCode\2019_py3\q3p2.csv', header=None).iloc[0, :]);
xy1 = pd.DataFrame(get_coords(p1), columns=['x', 'y']);
xy2 = pd.DataFrame(get_coords(p2), columns=['x', 'y']);
xy1['t'] = np.arange(0, xy1.shape[0]);
xy2['t'] = np.arange(0, xy2.shape[0]);
merge = xy1.merge(xy2, how='inner', on=('x', 'y'));
dist = abs(merge.loc[1:, ('x', 'y')]).sum(axis=1);
time = abs(merge.loc[1:, ('t_x', 't_y')]).sum(axis=1);
print(dist.min());
print(time.min());

#%% Question 4
ll = 206938;
ul = 679128;

lls = [int(d) for d in str(ll)];
uls = [int(d) for d in str(ul)];

def int2list(num):
    return [int(s) for s in str(num)];
    
def list2int(array):
    return int("".join([str(n) for n in array]));

def check_num(num_list):
    flag1 = True;
    flag2 = False;
    flag3 = False;
    if [num_list.count(x) for x in range(10)].count(2) > 0:
        flag3 = True;
    for i in range(len(num_list) - 1):
        if num_list[i] > num_list[i + 1]:
            flag1 = False;
            break;
        if num_list[i] == num_list[i+1]:
            flag2 = True;
    return flag1 & flag2 & flag3;

def next_num(num_list):
    # set minimum value for next integer
    for i in range(len(num_list) - 1):
        if num_list[i + 1] < num_list[i]:
            for j in range(i+1, len(num_list)):
                num_list[j] = num_list[i];
            return(num_list);
    # add one smartly
    for j in range(len(num_list)-1, -1, -1):
        if num_list[j] == 9:
            num_list = int2list(list2int(num_list) + 1);
            return num_list;
        else:
            num_list[j] = num_list[j] + 1;
            return(num_list);

def get_nums(start, end):
    curr_num = int2list(start);
    n = 0;
    while True:
        curr_num = next_num(curr_num);
        if list2int(curr_num) > end:
            return n;
        elif check_num(curr_num):
            print(curr_num);
            n = n+1;
    
n = get_nums(ll, ul);
print(n)
        
        
# t = next_num(lls.copy())

#%%
def int2list(num):
    return [int(s) for s in str(num)];

def list2int(array):
    return int("".join([str(n) for n in array]));

def get_instruction(inst):
    inst = int2list(inst);
    while len(inst) < 5:
        inst.insert(0, 0);
    mode = [inst[2], inst[1], inst[0]];
    inst = list2int(inst[3:]);
    return inst, mode;

def get_val_pos(x, i, mode):
    val = ['.', '.', '.'];
    pos = ['.', '.', '.'];
    for j in range(len(mode)):
        if mode[j] == 0:
            try:
                pos[j] = x[i+j+1];
            except:
                1
            try:
                val[j] = x[pos[j]];
            except:
                1
        elif mode[j] == 1:
            try:
                val[j] = x[i+j+1];
            except:
                1
    return val, pos;

def run_intcode2(code):
    
    i = 0;
    x = code.copy();
    
    while True:
        
        inst = x[i];
        
        if inst == 99:
            break;
        else:
            inst, mode = get_instruction(inst);
            val, pos = get_val_pos(x, i, mode);            

        if inst == 1:
            x[pos[2]] = val[0]+val[1];
            i = i + 4;
        elif inst == 2:
            x[pos[2]] = val[0]*val[1];
            i = i + 4;
        elif inst == 3:
            x[pos[0]] = int(input('what is the input? '));
            i = i + 2;
        elif inst == 4:
            print(val[0]);
            i = i + 2;
        elif inst == 5:
            if val[0] != 0:
                i = val[1];
            else:
                i = i + 3;
        elif inst == 6:
            if val[0] == 0:
                i = val[1];
            else:
                i = i + 3;
        elif inst == 7:
            if val[0] < val[1]:
                x[pos[2]] = 1;
            else:
                x[pos[2]] = 0;
            i = i + 4;
        elif inst == 8:
            if val[0] == val[1]:
                x[pos[2]] = 1;
            else:
                x[pos[2]] = 0;
            i = i + 4;
        else: 
            print('error');            
            
    return x;

x = [3,225,1,225,6,6,1100,1,238,225,104,0,1002,148,28,224,1001,224,-672,224,4,224,1002,223,8,223,101,3,224,224,1,224,223,223,1102,8,21,225,1102,13,10,225,1102,21,10,225,1102,6,14,225,1102,94,17,225,1,40,173,224,1001,224,-90,224,4,224,102,8,223,223,1001,224,4,224,1,224,223,223,2,35,44,224,101,-80,224,224,4,224,102,8,223,223,101,6,224,224,1,223,224,223,1101,26,94,224,101,-120,224,224,4,224,102,8,223,223,1001,224,7,224,1,224,223,223,1001,52,70,224,101,-87,224,224,4,224,1002,223,8,223,1001,224,2,224,1,223,224,223,1101,16,92,225,1101,59,24,225,102,83,48,224,101,-1162,224,224,4,224,102,8,223,223,101,4,224,224,1,223,224,223,1101,80,10,225,101,5,143,224,1001,224,-21,224,4,224,1002,223,8,223,1001,224,6,224,1,223,224,223,1102,94,67,224,101,-6298,224,224,4,224,102,8,223,223,1001,224,3,224,1,224,223,223,4,223,99,0,0,0,677,0,0,0,0,0,0,0,0,0,0,0,1105,0,99999,1105,227,247,1105,1,99999,1005,227,99999,1005,0,256,1105,1,99999,1106,227,99999,1106,0,265,1105,1,99999,1006,0,99999,1006,227,274,1105,1,99999,1105,1,280,1105,1,99999,1,225,225,225,1101,294,0,0,105,1,0,1105,1,99999,1106,0,300,1105,1,99999,1,225,225,225,1101,314,0,0,106,0,0,1105,1,99999,108,677,677,224,102,2,223,223,1005,224,329,101,1,223,223,1107,677,226,224,102,2,223,223,1006,224,344,101,1,223,223,1107,226,226,224,102,2,223,223,1006,224,359,101,1,223,223,1108,677,677,224,102,2,223,223,1005,224,374,101,1,223,223,8,677,226,224,1002,223,2,223,1005,224,389,101,1,223,223,108,226,677,224,1002,223,2,223,1006,224,404,1001,223,1,223,107,677,677,224,102,2,223,223,1006,224,419,101,1,223,223,1007,226,226,224,102,2,223,223,1005,224,434,101,1,223,223,1007,677,677,224,102,2,223,223,1005,224,449,1001,223,1,223,8,677,677,224,1002,223,2,223,1006,224,464,101,1,223,223,1108,677,226,224,1002,223,2,223,1005,224,479,101,1,223,223,7,677,226,224,1002,223,2,223,1005,224,494,101,1,223,223,1008,677,677,224,1002,223,2,223,1006,224,509,1001,223,1,223,1007,226,677,224,1002,223,2,223,1006,224,524,1001,223,1,223,107,226,226,224,1002,223,2,223,1006,224,539,1001,223,1,223,1107,226,677,224,102,2,223,223,1005,224,554,101,1,223,223,1108,226,677,224,102,2,223,223,1006,224,569,101,1,223,223,108,226,226,224,1002,223,2,223,1006,224,584,1001,223,1,223,7,226,226,224,1002,223,2,223,1006,224,599,101,1,223,223,8,226,677,224,102,2,223,223,1005,224,614,101,1,223,223,7,226,677,224,1002,223,2,223,1005,224,629,101,1,223,223,1008,226,677,224,1002,223,2,223,1006,224,644,101,1,223,223,107,226,677,224,1002,223,2,223,1005,224,659,1001,223,1,223,1008,226,226,224,1002,223,2,223,1006,224,674,1001,223,1,223,4,223,99,226];
y = x.copy();
y = run_intcode2(y);

#%%
# orbits = pd.read_csv(r'C:\Users\Leonard\Documents\GitHub\AdventOfCode\2019_py3\q6_test.csv', header=None)
orbits = pd.read_csv(r'C:\Users\Leonard\Documents\GitHub\AdventOfCode\2019_py3\q6.csv', header=None)
orbits = orbits[0].str.split(')', expand=True).rename(columns={0:'in', 1:'out'});
objects = np.unique(np.append(orbits.loc[:, 'in'].unique(), orbits.loc[:, 'out'].unique()));
objects = pd.DataFrame({'ID': np.arange(len(objects)), 'obj': objects});
orbits = orbits.merge(objects, how='left', left_on='in', right_on='obj').rename(columns={'ID': 'in_ID'}).drop(columns='obj')
orbits = orbits.merge(objects, how='left', left_on='out', right_on='obj').rename(columns={'ID': 'out_ID'}).drop(columns='obj')
adj_mat = sp.coo_matrix((np.ones(orbits['in_ID'].shape), (orbits['out_ID'], orbits['in_ID'])), shape=(len(objects), len(objects)))
idx = (objects['obj']=='YOU') | (objects['obj']=='SAN')

dist_mat = sp.csgraph.floyd_warshall(sp.csc_matrix(adj_mat), directed=True);
dist_mat[np.isinf(dist_mat)] = np.nan;
print(np.sum(dist_mat > 0))

adj_mat2 = adj_mat.todense() + adj_mat.transpose().todense();
dist_mat = sp.csgraph.floyd_warshall(sp.csc_matrix(adj_mat2), directed=True);
print(dist_mat[idx][:, idx])

#%%

def int2list(num):
    return [int(s) for s in str(num)];

def list2int(array):
    return int("".join([str(n) for n in array]));

def get_instruction(inst):
    inst = int2list(inst);
    while len(inst) < 5:
        inst.insert(0, 0);
    mode = [inst[2], inst[1], inst[0]];
    inst = list2int(inst[3:]);
    return inst, mode;

def get_val_pos(x, i, mode):
    val = ['.', '.', '.'];
    pos = ['.', '.', '.'];
    for j in range(len(mode)):
        if mode[j] == 0:
            try:
                pos[j] = x[i+j+1];
            except:
                1
            try:
                val[j] = x[pos[j]];
            except:
                1
        elif mode[j] == 1:
            try:
                val[j] = x[i+j+1];
            except:
                1
    return val, pos;

def run_intcode3(code=[], code_input=[], code_idx=0, code_input_idx=0):
    
    code_output = np.array([]);
    
    while True:
        
        inst = code[code_idx];
        
        if inst == 99:
            break;
        else:
            inst, mode = get_instruction(inst);
            val, pos = get_val_pos(code, code_idx, mode);            

        if inst == 1:
            code[pos[2]] = val[0] + val[1];
            code_idx += 4;
            
        elif inst == 2:
            code[pos[2]] = val[0] * val[1];
            code_idx += 4;
            
        elif inst == 3:
            if len(code_input) == code_input_idx:
                break;
            else:
                curr_input = code_input[code_input_idx]
                code[pos[0]] = curr_input;
                code_idx += 2;
                code_input_idx += 1;
            
        elif inst == 4:
            code_output = np.append(code_output, val[0]);
            code_idx += 2;
            
        elif inst == 5:
            if val[0] != 0:
                code_idx = val[1];
            else:
                code_idx += 3;
                
        elif inst == 6:
            if val[0] == 0:
                code_idx = val[1];
            else:
                code_idx += 3;
                
        elif inst == 7:
            if val[0] < val[1]:
                code[pos[2]] = 1;
            else:
                code[pos[2]] = 0;
            code_idx += 4;
            
        elif inst == 8:
            if val[0] == val[1]:
                code[pos[2]] = 1;
            else:
                code[pos[2]] = 0;
            code_idx += 4;
            
        else: 
            print('error');            
      
    return [code, code_idx, code_input_idx, inst], code_output;


x = np.array([3,8,1001,8,10,8,105,1,0,0,21,38,63,88,97,118,199,280,361,442,99999,3,9,1002,9,3,9,101,2,9,9,1002,9,4,9,4,9,99,3,9,101,3,9,9,102,5,9,9,101,3,9,9,1002,9,3,9,101,3,9,9,4,9,99,3,9,1002,9,2,9,1001,9,3,9,102,3,9,9,101,2,9,9,1002,9,4,9,4,9,99,3,9,102,2,9,9,4,9,99,3,9,102,4,9,9,101,5,9,9,102,2,9,9,101,5,9,9,4,9,99,3,9,1002,9,2,9,4,9,3,9,101,1,9,9,4,9,3,9,102,2,9,9,4,9,3,9,101,1,9,9,4,9,3,9,101,2,9,9,4,9,3,9,1001,9,2,9,4,9,3,9,102,2,9,9,4,9,3,9,102,2,9,9,4,9,3,9,101,1,9,9,4,9,3,9,102,2,9,9,4,9,99,3,9,101,1,9,9,4,9,3,9,102,2,9,9,4,9,3,9,1001,9,2,9,4,9,3,9,1001,9,2,9,4,9,3,9,1001,9,1,9,4,9,3,9,1001,9,1,9,4,9,3,9,1001,9,2,9,4,9,3,9,102,2,9,9,4,9,3,9,1002,9,2,9,4,9,3,9,1001,9,1,9,4,9,99,3,9,1002,9,2,9,4,9,3,9,1002,9,2,9,4,9,3,9,102,2,9,9,4,9,3,9,1002,9,2,9,4,9,3,9,1001,9,1,9,4,9,3,9,102,2,9,9,4,9,3,9,102,2,9,9,4,9,3,9,101,1,9,9,4,9,3,9,102,2,9,9,4,9,3,9,102,2,9,9,4,9,99,3,9,102,2,9,9,4,9,3,9,101,1,9,9,4,9,3,9,1002,9,2,9,4,9,3,9,1002,9,2,9,4,9,3,9,102,2,9,9,4,9,3,9,1002,9,2,9,4,9,3,9,1001,9,2,9,4,9,3,9,101,2,9,9,4,9,3,9,1001,9,2,9,4,9,3,9,101,1,9,9,4,9,99,3,9,101,1,9,9,4,9,3,9,101,1,9,9,4,9,3,9,101,1,9,9,4,9,3,9,102,2,9,9,4,9,3,9,1001,9,1,9,4,9,3,9,1001,9,2,9,4,9,3,9,101,2,9,9,4,9,3,9,102,2,9,9,4,9,3,9,1001,9,1,9,4,9,3,9,1001,9,2,9,4,9,99]);

code_output = np.zeros(len(perms));
phase_input_list = list(itertools.permutations(range(5)));
for i in range(len(phase_input_list)):
    phase_input = phase_input_list[i];
    x_output = 0;
    for j in range(5):
        state, x_output = run_intcode3(
            code = x, 
            code_input = np.append(phase_input[j], x_output)
            );
    code_output[i] = x_output;
print(np.max(code_output));

code_output = np.zeros(len(perms));
phase_input_list = list(itertools.permutations(np.arange(5) + 5));
output_list = np.zeros(len(phase_input_list));

for i in range(len(phase_input_list)):
    
    phase_input = phase_input_list[i];
    code_list = [np.array([])]*5; 
    input_list = [np.array([])]*5;
    curr_loc = [0]*5;
    curr_input_loc = [0]*5;
    curr_inst = np.zeros(len(curr_loc));
    x_output = 0;
    curr_code = 0;
    for j in range(len(input_list)):
        code_list[j] = x.copy();
        input_list[j] = np.append(input_list[j], phase_input[j]);
    input_list[0] = np.append(input_list[0], 0);
        
    while True:
        state, x_output = run_intcode3(
            code = code_list[curr_code], 
            code_input = input_list[curr_code],
            code_idx = curr_loc[curr_code],
            code_input_idx = curr_input_loc[curr_code]
            );
        code_list[curr_code] = state[0];
        curr_loc[curr_code] = state[1];
        curr_input_loc[curr_code] = state[2];
        curr_inst[curr_code] = state[3];
        curr_code = np.mod(curr_code + 1, len(phase_input));
        input_list[curr_code] = np.append(input_list[curr_code], x_output);
        
        if np.sum(curr_inst == 99) == len(code_list) :
            break;
    
    output_list[i] = x_output;
          
print(np.max(output_list));

#%% day 9

def int2list(num):
    return [int(s) for s in str(num)];

def list2int(array):
    return int("".join([str(n) for n in array]));

def get_instruction(inst):
    inst = int2list(inst);
    while len(inst) < 5:
        inst.insert(0, 0);
    mode = [inst[2], inst[1], inst[0]];
    inst = list2int(inst[3:]);
    return inst, mode;

def get_val_pos(x, i, inst, mode, relative_base):
        
    if inst in [1, 2, 7, 8]:
        max_j = 3;
    elif inst in [5, 6, 9]:
        max_j = 2;
    elif inst in [3, 4]:
        max_j = 1;
        
    val = ['.']*max_j;
    pos = ['.']*max_j;
        
    for j in range(max_j):
        
        if mode[j] == 0:
            pos[j] = x[i+j+1];
            
            if pos[j] >= len(x):
                x = np.append(x, np.zeros(pos[j] - len(x) + 1).astype(int));
            val[j] = x[pos[j]];
                
        elif mode[j] == 1:
            val[j] = x[i+j+1];
            
        elif mode[j] == 2:
            pos[j] = int(x[i+j+1] + relative_base);
            if pos[j] >= len(x):
                x = np.append(x, int(np.zeros(pos[j] - len(x) + 1)));
            val[j] = x[pos[j]];
            
    return x, val, pos;

def run_intcode4(code=[], code_input=[], code_idx=0, code_input_idx=0, relative_base=0):
    
    code_output = np.array([]);
    
    while True:
        
        inst = code[code_idx];
        
        if inst == 99:
            break;
        else:
            inst, mode = get_instruction(inst);
            code, val, pos = get_val_pos(code, code_idx, inst, mode, relative_base);      
        
        if inst == 1:
            code[pos[2]] = val[0] + val[1];
            code_idx += 4;
            
        elif inst == 2:
            code[pos[2]] = val[0] * val[1];
            code_idx += 4;
            
        elif inst == 3:
            if len(code_input) == code_input_idx:
                break;
            else:
                curr_input = code_input[code_input_idx]
                code[pos[0]] = curr_input;
                code_idx += 2;
                code_input_idx += 1;
            
        elif inst == 4:
            code_output = np.append(code_output, val[0]);
            code_idx += 2;
            
        elif inst == 5:
            if val[0] != 0:
                code_idx = val[1];
            else:
                code_idx += 3;
                
        elif inst == 6:
            if val[0] == 0:
                code_idx = val[1];
            else:
                code_idx += 3;
                
        elif inst == 7:
            if val[0] < val[1]:
                code[pos[2]] = 1;
            else:
                code[pos[2]] = 0;
            code_idx += 4;
            
        elif inst == 8:
            if val[0] == val[1]:
                code[pos[2]] = 1;
            else:
                code[pos[2]] = 0;
            code_idx += 4;
            
        elif inst == 9:
            relative_base = relative_base + val[0];
            code_idx += 2;
            
        else: 
            print('error');            
      
        # print(inst, val, pos, )
        
    return [code, code_idx, code_input_idx, inst], code_output;

# x = np.array([109,1,204,-1,1001,100,1,100,1008,100,16,101,1006,101,0,99], dtype=object);
# x = np.array([1102,34915192,34915192,7,4,7,99,0], dtype=object);
# x = np.array([104,999999,99], dtype=object);
x = np.array([1102,34463338,34463338,63,1007,63,34463338,63,1005,63,53,1102,1,3,1000,109,988,209,12,9,1000,209,6,209,3,203,0,1008,1000,1,63,1005,63,65,1008,1000,2,63,1005,63,902,1008,1000,0,63,1005,63,58,4,25,104,0,99,4,0,104,0,99,4,17,104,0,99,0,0,1101,0,39,1005,1102,1,1,1021,1101,0,212,1025,1101,0,24,1014,1102,22,1,1019,1101,0,35,1003,1101,38,0,1002,1101,0,571,1026,1102,32,1,1006,1102,31,1,1000,1102,25,1,1018,1102,1,37,1016,1101,0,820,1023,1102,1,29,1004,1101,564,0,1027,1101,0,375,1028,1101,26,0,1013,1102,1,370,1029,1101,21,0,1007,1101,0,0,1020,1102,1,30,1001,1102,36,1,1011,1102,1,27,1017,1101,0,28,1012,1101,0,217,1024,1101,823,0,1022,1102,1,20,1009,1101,0,23,1010,1101,34,0,1015,1101,33,0,1008,109,5,1208,0,39,63,1005,63,199,4,187,1106,0,203,1001,64,1,64,1002,64,2,64,109,13,2105,1,6,4,209,1105,1,221,1001,64,1,64,1002,64,2,64,109,-4,21108,40,39,-1,1005,1013,241,1001,64,1,64,1105,1,243,4,227,1002,64,2,64,109,5,21102,41,1,-1,1008,1018,40,63,1005,63,267,1001,64,1,64,1106,0,269,4,249,1002,64,2,64,109,-28,1202,10,1,63,1008,63,30,63,1005,63,291,4,275,1106,0,295,1001,64,1,64,1002,64,2,64,109,24,21107,42,43,-4,1005,1011,313,4,301,1106,0,317,1001,64,1,64,1002,64,2,64,109,-8,21108,43,43,3,1005,1010,335,4,323,1105,1,339,1001,64,1,64,1002,64,2,64,109,-8,1207,4,34,63,1005,63,359,1001,64,1,64,1106,0,361,4,345,1002,64,2,64,109,26,2106,0,3,4,367,1106,0,379,1001,64,1,64,1002,64,2,64,109,-21,2102,1,-2,63,1008,63,37,63,1005,63,399,1105,1,405,4,385,1001,64,1,64,1002,64,2,64,109,2,1207,-2,30,63,1005,63,427,4,411,1001,64,1,64,1105,1,427,1002,64,2,64,109,4,2108,36,-5,63,1005,63,447,1001,64,1,64,1106,0,449,4,433,1002,64,2,64,109,-13,1201,8,0,63,1008,63,41,63,1005,63,469,1106,0,475,4,455,1001,64,1,64,1002,64,2,64,109,14,21107,44,43,3,1005,1014,495,1001,64,1,64,1106,0,497,4,481,1002,64,2,64,109,2,1205,8,511,4,503,1106,0,515,1001,64,1,64,1002,64,2,64,109,14,1206,-6,527,1105,1,533,4,521,1001,64,1,64,1002,64,2,64,109,-29,2107,31,8,63,1005,63,551,4,539,1105,1,555,1001,64,1,64,1002,64,2,64,109,28,2106,0,1,1001,64,1,64,1106,0,573,4,561,1002,64,2,64,109,-3,21101,45,0,-4,1008,1019,45,63,1005,63,595,4,579,1105,1,599,1001,64,1,64,1002,64,2,64,109,-23,1208,2,39,63,1005,63,615,1105,1,621,4,605,1001,64,1,64,1002,64,2,64,109,15,2108,32,-9,63,1005,63,643,4,627,1001,64,1,64,1105,1,643,1002,64,2,64,109,-9,2107,33,0,63,1005,63,659,1106,0,665,4,649,1001,64,1,64,1002,64,2,64,109,7,21101,46,0,2,1008,1015,49,63,1005,63,689,1001,64,1,64,1106,0,691,4,671,1002,64,2,64,109,-8,2101,0,-3,63,1008,63,35,63,1005,63,711,1105,1,717,4,697,1001,64,1,64,1002,64,2,64,109,12,1202,-9,1,63,1008,63,31,63,1005,63,741,1001,64,1,64,1105,1,743,4,723,1002,64,2,64,109,-27,2102,1,10,63,1008,63,31,63,1005,63,769,4,749,1001,64,1,64,1105,1,769,1002,64,2,64,109,9,2101,0,1,63,1008,63,31,63,1005,63,791,4,775,1106,0,795,1001,64,1,64,1002,64,2,64,109,28,1206,-7,809,4,801,1105,1,813,1001,64,1,64,1002,64,2,64,2105,1,-4,1106,0,829,4,817,1001,64,1,64,1002,64,2,64,109,-15,21102,47,1,-2,1008,1010,47,63,1005,63,851,4,835,1106,0,855,1001,64,1,64,1002,64,2,64,109,5,1205,3,867,1106,0,873,4,861,1001,64,1,64,1002,64,2,64,109,-12,1201,0,0,63,1008,63,39,63,1005,63,895,4,879,1105,1,899,1001,64,1,64,4,64,99,21101,0,27,1,21102,913,1,0,1106,0,920,21201,1,47951,1,204,1,99,109,3,1207,-2,3,63,1005,63,962,21201,-2,-1,1,21101,0,940,0,1105,1,920,21201,1,0,-1,21201,-2,-3,1,21101,0,955,0,1106,0,920,22201,1,-1,-2,1105,1,966,21202,-2,1,-2,109,-3,2105,1,0], dtype=object);

code = x.copy();
t1, t2 = run_intcode4(code=code, code_input=np.array([1]))
print(t2[0])
t1, t2 = run_intcode4(code=code, code_input=np.array([2]))
print(t2[0])

#%% day12

# init_pos = np.array([[-1, 0, 2], [2, -10, -7], [4, -8, 8], [3, 5, -1]]);

# init_pos = np.array([[-8, -10, 0], [5, 5, 10], [2, -7, 3], [9, -8, -3]]);

init_pos = np.array([[-2, 9, -5], [16, 19, 9], [0, 3, 6], [11, 0, 11]]);

init_vel = np.zeros(pos.shape);
n_steps = 100;
p_energy = np.zeros(n_steps);
k_energy = np.zeros(n_steps)

pos = init_pos.copy();
vel = init_vel.copy();
def next_step(pos, vel):
    for p in pos:
        vel = vel + 1*((p - pos) > 0);
        vel = vel - 1*((p - pos) < 0);
    return pos+vel, vel;

def get_energy(pos, vel):
    p = np.sum(np.abs(pos), axis=1);
    k = np.sum(np.abs(vel), axis=1);
    t = np.sum(p*k);
    return p, k, t;

for i in range(1000):
    pos, vel = next_step(pos, vel);
    p, k, t = get_energy(pos, vel);
print(t)

x = init_pos[:, 0];
vx = init_vel[:, 0];
y = init_pos[:, 1];
vy = init_vel[:, 1];
z = init_pos[:, 2];
vz = init_vel[:, 2];

x_idx = 0;
while True:
    x_idx += 1;
    x, vx = next_step(x, vx);
    if np.array_equal(x, init_pos[:, 0]):
        if np.array_equal(vx, init_vel[:, 0]):
            break;
y_idx = 0;
while True:
    y_idx += 1;
    y, vy = next_step(y, vy);
    if np.array_equal(y, init_pos[:, 1]):
        if np.array_equal(vy, init_vel[:, 1]):
            break;
z_idx = 0;
while True:
    z_idx += 1;
    z, vz = next_step(z, vz);
    if np.array_equal(z, init_pos[:, 2]):
        if np.array_equal(vz, init_vel[:, 2]):
            break;
            
print(np.lcm(np.lcm(x_idx, y_idx, dtype=object), z_idx, dtype=object))
            
#%% day 13

def int2list(num):
    return [int(s) for s in str(num)];

def list2int(array):
    return int("".join([str(n) for n in array]));

def get_instruction(inst):
    inst = int2list(inst);
    while len(inst) < 5:
        inst.insert(0, 0);
    mode = [inst[2], inst[1], inst[0]];
    inst = list2int(inst[3:]);
    return inst, mode;

def get_val_pos(x, i, inst, mode, relative_base):
        
    if inst in [1, 2, 7, 8]:
        max_j = 3;
    elif inst in [5, 6, 9]:
        max_j = 2;
    elif inst in [3, 4]:
        max_j = 1;
        
    val = ['.']*max_j;
    pos = ['.']*max_j;
        
    for j in range(max_j):
        
        if mode[j] == 0:
            pos[j] = x[i+j+1];
            
            if pos[j] >= len(x):
                x = np.append(x, np.zeros(pos[j] - len(x) + 1).astype(int));
            val[j] = x[pos[j]];
                
        elif mode[j] == 1:
            val[j] = x[i+j+1];
            
        elif mode[j] == 2:
            pos[j] = int(x[i+j+1] + relative_base);
            if pos[j] >= len(x):
                x = np.append(x, np.zeros(pos[j] - len(x) + 1).astype(int));
            val[j] = x[pos[j]];
            
    return x, val, pos;

def run_intcode5(code=[], code_idx=0, relative_base=0, flag=0, curr_input=9):
    
    code_output = np.array([]);
    
    while True:
        
        inst = code[code_idx];
        
        if inst == 99:
            break;
        else:
            inst, mode = get_instruction(inst);
            code, val, pos = get_val_pos(code, code_idx, inst, mode, relative_base);      
        
        if inst == 1:
            code[pos[2]] = val[0] + val[1];
            code_idx += 4;
            
        elif inst == 2:
            code[pos[2]] = val[0] * val[1];
            code_idx += 4;
            
        elif inst == 3:
            if flag == 0:
                return [code, code_idx, inst], code_output;
            elif flag == 1:
                flag = 0;
                if curr_input == 9:
                    while True:
                        try:
                            curr_input = int(input('input? '));
                            break;
                        except:
                            1
                if curr_input == 1:
                    curr_input = 1;
                if curr_input == 3:
                    curr_input = -1;
                if curr_input == 2:
                    curr_input = 0;
                code[pos[0]] = curr_input;
                code_idx += 2;                
            
        elif inst == 4:
            code_output = np.append(code_output, val[0]);
            code_idx += 2;
            
        elif inst == 5:
            if val[0] != 0:
                code_idx = val[1];
            else:
                code_idx += 3;
                
        elif inst == 6:
            if val[0] == 0:
                code_idx = val[1];
            else:
                code_idx += 3;
                
        elif inst == 7:
            if val[0] < val[1]:
                code[pos[2]] = 1;
            else:
                code[pos[2]] = 0;
            code_idx += 4;
            
        elif inst == 8:
            if val[0] == val[1]:
                code[pos[2]] = 1;
            else:
                code[pos[2]] = 0;
            code_idx += 4;
            
        elif inst == 9:
            relative_base = relative_base + val[0];
            code_idx += 2;
            
        else: 
            print('error');            
              
    return [code, code_idx, inst], code_output;

def coords2im(coords, score=0):
    im = np.zeros(np.max(coords[:, :-1], axis=0).astype(int)+1);
    for i in range(coords.shape[0]):
        if coords[i, 0] == -1:
            score = coords[i, 2];
        else:
            im[coords[i, 0], coords[i, 1]] = coords[i, 2];
    return im, score;

x = np.array([1,380,379,385,1008,2751,751761,381,1005,381,12,99,109,2752,1101,0,0,383,1101,0,0,382,21001,382,0,1,21002,383,1,2,21101,37,0,0,1106,0,578,4,382,4,383,204,1,1001,382,1,382,1007,382,44,381,1005,381,22,1001,383,1,383,1007,383,24,381,1005,381,18,1006,385,69,99,104,-1,104,0,4,386,3,384,1007,384,0,381,1005,381,94,107,0,384,381,1005,381,108,1105,1,161,107,1,392,381,1006,381,161,1101,0,-1,384,1106,0,119,1007,392,42,381,1006,381,161,1102,1,1,384,20102,1,392,1,21101,22,0,2,21101,0,0,3,21102,1,138,0,1106,0,549,1,392,384,392,20101,0,392,1,21102,22,1,2,21102,3,1,3,21101,0,161,0,1106,0,549,1101,0,0,384,20001,388,390,1,20102,1,389,2,21101,180,0,0,1105,1,578,1206,1,213,1208,1,2,381,1006,381,205,20001,388,390,1,21002,389,1,2,21102,1,205,0,1106,0,393,1002,390,-1,390,1102,1,1,384,20101,0,388,1,20001,389,391,2,21101,0,228,0,1105,1,578,1206,1,261,1208,1,2,381,1006,381,253,21002,388,1,1,20001,389,391,2,21102,253,1,0,1106,0,393,1002,391,-1,391,1102,1,1,384,1005,384,161,20001,388,390,1,20001,389,391,2,21101,0,279,0,1106,0,578,1206,1,316,1208,1,2,381,1006,381,304,20001,388,390,1,20001,389,391,2,21101,304,0,0,1105,1,393,1002,390,-1,390,1002,391,-1,391,1101,1,0,384,1005,384,161,20102,1,388,1,20102,1,389,2,21102,0,1,3,21101,0,338,0,1106,0,549,1,388,390,388,1,389,391,389,21002,388,1,1,20101,0,389,2,21101,4,0,3,21102,1,365,0,1106,0,549,1007,389,23,381,1005,381,75,104,-1,104,0,104,0,99,0,1,0,0,0,0,0,0,376,20,19,1,1,22,109,3,21201,-2,0,1,21202,-1,1,2,21101,0,0,3,21101,0,414,0,1106,0,549,22102,1,-2,1,21201,-1,0,2,21101,429,0,0,1105,1,601,2101,0,1,435,1,386,0,386,104,-1,104,0,4,386,1001,387,-1,387,1005,387,451,99,109,-3,2105,1,0,109,8,22202,-7,-6,-3,22201,-3,-5,-3,21202,-4,64,-2,2207,-3,-2,381,1005,381,492,21202,-2,-1,-1,22201,-3,-1,-3,2207,-3,-2,381,1006,381,481,21202,-4,8,-2,2207,-3,-2,381,1005,381,518,21202,-2,-1,-1,22201,-3,-1,-3,2207,-3,-2,381,1006,381,507,2207,-3,-4,381,1005,381,540,21202,-4,-1,-1,22201,-3,-1,-3,2207,-3,-4,381,1006,381,529,22101,0,-3,-7,109,-8,2105,1,0,109,4,1202,-2,44,566,201,-3,566,566,101,639,566,566,1201,-1,0,0,204,-3,204,-2,204,-1,109,-4,2105,1,0,109,3,1202,-1,44,593,201,-2,593,593,101,639,593,593,21001,0,0,-2,109,-3,2106,0,0,109,3,22102,24,-2,1,22201,1,-1,1,21101,0,541,2,21102,750,1,3,21101,0,1056,4,21102,1,630,0,1105,1,456,21201,1,1695,-2,109,-3,2105,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,2,2,2,2,2,0,2,2,2,0,2,2,2,2,0,0,2,2,2,0,2,2,2,2,2,2,2,0,2,0,2,2,2,2,2,2,0,2,2,0,0,1,1,0,2,2,0,0,2,2,2,2,2,0,0,2,2,0,2,2,0,2,0,2,2,0,0,2,2,0,2,0,2,0,0,2,0,2,2,0,2,0,2,2,0,1,1,0,2,2,2,2,0,0,0,2,0,2,0,0,0,0,2,2,2,2,0,2,2,2,0,0,2,2,2,0,2,2,2,2,2,0,0,2,2,0,0,2,0,1,1,0,2,2,0,2,2,2,0,0,0,2,2,0,2,2,0,2,2,0,0,2,0,2,0,0,2,2,2,2,0,2,2,2,0,2,0,0,2,2,2,2,0,1,1,0,0,0,2,2,0,0,0,0,0,2,0,0,0,2,2,0,0,2,0,0,2,2,0,0,2,2,2,2,0,2,0,0,2,0,2,2,2,0,2,2,0,1,1,0,0,2,2,0,2,2,2,2,2,2,2,2,0,2,2,0,2,2,0,0,2,0,0,0,2,2,0,2,2,0,0,0,0,2,0,2,0,0,0,2,0,1,1,0,2,2,2,0,2,0,2,2,0,2,0,0,2,0,2,2,2,0,2,2,2,2,0,0,0,0,0,2,0,2,2,2,2,0,0,0,2,0,0,0,0,1,1,0,0,0,2,2,2,2,2,2,0,2,0,2,2,0,2,2,2,2,0,0,0,0,2,2,0,0,2,2,2,0,2,0,2,2,0,0,2,2,2,0,0,1,1,0,0,0,0,2,2,2,0,0,2,0,2,2,0,2,2,0,0,0,0,2,0,2,2,2,0,2,2,0,2,2,0,0,2,2,0,2,2,2,0,2,0,1,1,0,0,2,0,0,0,2,2,0,2,0,2,2,0,0,2,0,2,2,2,2,2,2,0,0,2,2,0,0,2,2,2,0,2,2,0,0,0,0,2,2,0,1,1,0,0,2,0,0,0,2,2,2,0,2,0,2,2,2,2,0,2,2,0,0,2,2,2,2,0,2,2,2,2,2,2,0,0,0,0,2,2,2,2,0,0,1,1,0,2,2,2,2,2,0,2,0,0,2,2,0,2,0,2,0,2,2,0,2,0,2,2,2,2,2,0,0,0,0,2,2,2,2,0,0,0,0,0,0,0,1,1,0,2,2,0,2,2,0,0,0,2,2,2,2,0,0,2,0,2,0,2,0,2,2,0,0,2,2,0,2,2,2,0,2,0,0,0,0,2,2,2,2,0,1,1,0,0,2,2,2,2,2,0,2,0,2,2,0,0,0,2,0,2,2,0,0,0,2,2,2,2,2,2,2,2,2,0,0,0,2,0,0,0,0,2,0,0,1,1,0,0,0,0,2,0,0,2,2,2,2,2,2,0,2,0,2,2,2,2,0,0,0,2,2,2,2,0,2,2,0,2,2,0,0,0,2,2,2,2,0,0,1,1,0,0,0,0,2,0,2,2,2,2,2,0,2,2,2,2,2,2,0,0,2,0,2,2,2,2,2,2,0,0,2,0,2,2,2,0,2,2,0,2,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,27,30,18,28,94,18,40,59,97,67,72,28,75,43,73,43,86,45,58,6,22,11,34,64,74,45,90,20,49,72,59,3,65,32,80,39,51,12,22,48,11,98,5,45,80,41,88,83,63,29,65,45,80,53,23,68,27,51,43,98,46,4,98,98,68,64,28,95,5,94,12,52,91,15,35,29,47,3,60,54,75,24,71,67,22,53,77,40,47,41,8,53,38,40,83,25,65,80,44,30,37,57,62,13,74,23,13,22,43,31,76,77,22,21,16,89,16,18,95,31,21,37,66,33,85,38,35,29,47,90,30,95,17,43,58,40,3,41,49,59,96,38,37,73,68,83,92,83,45,25,10,12,51,33,54,53,67,95,46,58,97,19,4,51,5,46,42,49,65,67,88,18,64,84,19,72,9,80,71,34,92,76,62,86,68,19,87,88,42,65,40,91,45,44,56,58,50,53,98,87,62,97,27,60,16,9,19,9,66,85,58,71,42,63,90,24,86,12,37,27,84,87,79,16,4,90,98,13,17,83,87,24,32,19,60,46,77,69,55,38,68,92,36,13,61,27,76,76,73,32,13,78,89,38,93,32,30,66,67,31,39,8,79,57,42,84,11,56,24,77,57,84,80,75,97,69,83,61,69,69,19,22,82,7,35,85,58,88,24,91,91,12,92,28,43,68,8,17,67,65,20,65,72,66,94,93,11,85,27,72,11,26,42,25,83,11,44,53,22,51,32,27,16,67,74,39,37,51,13,78,54,82,14,64,62,49,89,68,60,25,51,32,50,5,28,78,18,26,9,61,41,7,97,48,25,16,29,50,1,40,91,46,87,69,51,72,84,95,20,78,49,20,5,70,30,72,15,36,46,43,81,33,2,67,10,33,68,59,28,46,23,64,23,58,8,14,47,37,82,50,3,63,15,85,24,75,5,35,63,36,17,70,30,35,24,35,81,26,97,24,65,42,26,37,69,67,52,9,79,91,56,53,96,75,78,34,98,47,67,82,34,96,67,75,55,12,56,50,32,56,3,25,38,70,81,24,44,50,69,51,25,66,54,14,71,54,79,88,62,43,92,38,23,61,31,29,78,9,56,2,61,15,58,73,5,97,47,81,84,39,77,81,52,63,30,91,81,74,27,20,98,4,53,32,95,11,13,28,91,97,45,67,12,65,78,41,18,30,98,69,88,58,14,55,42,6,64,14,55,98,22,16,51,4,16,89,96,21,38,2,8,49,70,11,94,34,19,5,98,25,27,42,82,67,80,67,22,78,50,18,67,55,92,61,43,66,11,25,73,53,8,79,38,81,84,60,89,14,33,18,86,78,55,96,92,6,36,64,96,50,64,93,20,3,27,79,98,53,69,77,85,62,68,83,67,71,29,68,52,71,98,31,17,75,9,43,92,39,19,58,97,64,70,58,74,10,37,74,28,35,97,33,21,27,72,72,82,77,91,89,21,52,76,82,24,91,73,31,19,90,97,37,5,88,53,7,20,89,72,20,2,28,61,68,40,17,81,27,92,78,11,30,78,62,98,15,38,7,46,21,48,81,43,1,70,70,26,20,37,91,28,40,81,53,90,54,10,92,88,98,13,94,88,41,66,31,69,45,28,64,77,24,71,11,11,56,93,65,5,57,54,93,7,43,6,96,1,22,36,15,67,88,33,70,14,46,71,12,57,37,80,46,13,53,63,77,61,56,3,12,60,34,77,70,56,57,5,83,38,9,70,32,79,90,85,50,65,5,45,64,29,47,15,2,46,30,13,89,53,19,80,38,63,25,10,46,94,93,86,61,41,22,98,52,81,76,85,34,25,72,26,64,44,52,47,69,21,39,67,35,43,75,21,58,3,15,71,44,77,42,20,67,17,25,12,6,50,2,63,78,41,80,26,19,9,30,36,16,86,63,51,7,29,16,5,94,15,53,26,69,67,21,38,13,65,78,34,94,58,25,33,14,12,57,67,96,18,79,37,64,83,23,59,23,52,13,50,88,98,26,11,85,39,36,47,10,77,4,81,25,6,14,11,45,72,70,94,2,54,23,83,95,58,20,25,15,24,69,35,96,70,93,79,79,5,39,83,43,29,4,64,82,52,16,84,36,89,31,21,90,41,39,23,35,83,65,89,53,6,64,68,55,59,57,17,78,92,6,17,1,84,86,19,78,69,34,12,36,41,60,16,37,24,31,31,91,13,93,38,17,80,25,37,9,49,59,96,80,68,64,40,35,45,10,16,13,23,33,52,63,84,9,93,31,40,70,69,19,22,79,25,20,47,83,40,29,86,96,84,23,31,42,82,87,83,5,70,25,15,23,77,41,31,73,2,3,74,69,44,31,10,96,52,93,88,98,56,11,55,47,34,86,63,7,11,86,77,77,39,75,44,31,58,10,20,1,751761], dtype=object);

code = x.copy();
t1, t2 = run_intcode5(code=code)
coords = np.reshape(t2, [-1, 3]);
im, score = coords2im(coords.astype(int));
print(np.sum(im==2));


code = x.copy();
code[0] = 2;
fig, ax = plt.subplots();
ax_im = ax.imshow(im);

# generate initial output
state, t2 = run_intcode5(code=code, flag=0);
    
while True:
    
    # see the output
    coords = np.reshape(t2, [-1, 3]);
    im, score = coords2im(coords.astype(int));
    
    print(score)
    # plt.imshow(im)
    # plt.draw()
    # fig.canvas.draw_idle();
    # plt.show();
    
    # process the state
    code = state[0];
    code_idx = state[1];
    inst = state[2];    
    if inst == 99:
        break;
        
    # find ball and paddle
    x_ball = np.where(im==4)[0];
    x_paddle = np.where(im==3)[0];
    if x_ball > x_paddle:
        next_input = 1;
    elif x_ball < x_paddle:
        next_input = 3;
    else:
        next_input = 2;

    # choose the next input
    state, t2 = run_intcode5(code=code, curr_input = next_input, flag=1);
    
print('final score: ', score);
    

