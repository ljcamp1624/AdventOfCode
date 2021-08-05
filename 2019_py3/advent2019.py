#%% Advent of code 2019
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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








