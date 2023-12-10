#%%
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import copy

#%% tools
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

def get_score(s):
    if s in ['A','X']:
        return 1
    elif s in ['B','Y']:
        return 2
    elif s in ['C','Z']:
        return 3
    
def get_score2(s):
    if s in ['X']:
        return 0
    elif s in ['Y']:
        return 3
    elif s in ['Z']:
        return 6

def resolve_game(s1, s2):
    if s1 == s2:
        return 3
    elif np.mod(s2,3) == np.mod(s1+1,3):
        return 6
    elif np.mod(s2,3) == np.mod(s1-1,3):
        return 0
    
def invert_game(s1, outcome):
    if outcome == 6:
        return np.mod(s1,3)+1
    elif outcome == 3:
        return s1
    elif outcome == 0:
        return np.mod(np.mod(s1,3)+1,3)+1

def matching_letter(s1, s2):
    s = []
    for i in s1:
        for j in s2:
            if i==j:
                s.append(i)
    return s

def get_priority(s):
    if ord(s) in range(65,91): # upper case letters
        return int(ord(s))-65+27
    elif ord(s) in range(97,123): # lower case letters
        return int(ord(s))-97+1

def parse_crates(crates):
    cols = []
    for s, i in zip(crates[-1], range(len(crates[-1]))):
        if s != ' ':
            cols.append(i)
    stacks = []
    for i in range(len(cols)):
        stacks.append([])
    for c, i in zip(cols, range(len(cols))):
        for row in crates[-2::-1]:
            if row[c] != ' ':
                stacks[i].append(row[c])
    return stacks

def multidir_max(mat, i, j):
    
    out = False
    v = -1
    y = j
    t1 = 0
    tree = False
    #up
    for x in range(i-1, -1, -1):
        v = max([v, int(mat[x][y])])
        if tree:
            continue
        elif int(mat[i][j]) > v:
            t1 += 1
        elif int(mat[i][j]) <= v:
            t1 += 1
            tree = True
    if int(mat[i][j]) > int(v):
        out = True
    
    v = -1
    t2 = 0
    tree = False
    #down
    for x in range(i+1, len(mat)):
        v = max([v, int(mat[x][y])])
        if tree:
            continue
        elif int(mat[i][j]) > v:
            t2 += 1
        elif int(mat[i][j]) <= v:
            t2 += 1
            tree = True
    if int(mat[i][j]) > int(v):
        out = True
    
    v = -1
    x = i
    t3 = 0
    tree = False
    for y in range(j-1, -1, -1):
        v = max([v, int(mat[x][y])])
        if tree:
            continue
        elif int(mat[i][j]) > v:
            t3 += 1
        elif int(mat[i][j]) <= v:
            t3 += 1
            tree = True
    if int(mat[i][j]) > int(v):
        out = True
    
    v = -1
    t4 = 0
    tree = False
    for y in range(j+1,len(mat[0])):
        v = max([v, int(mat[x][y])])
        if tree:
            continue
        elif int(mat[i][j]) > v:
            t4 += 1
        elif int(mat[i][j]) <= v:
            t4 += 1
            tree = True
    if int(mat[i][j]) > int(v):
        out = True
    
    return out, t1*t2*t3*t4

def parse_moves(moves):
    move_list = []
    for s in moves:
        move_list.append([int(x) for x in s.split(' ')[1::2]])
    return move_list

def move_head(head_pos, dir):
    if dir == 'U':
        head_pos[1] += 1
    elif dir == 'D':
        head_pos[1] += -1
    elif dir == 'R':
        head_pos[0] += 1
    elif dir == 'L':
        head_pos[0] += -1
    return head_pos

def move_tail(head_pos, tail_pos):
    diff = []
    for i in [0, 1]:
        diff.append(head_pos[i] - tail_pos[i]) 
    if max([abs(d) for d in diff])> 1:
        for d, i in zip(diff, range(len(diff))):
            tail_pos[i] += np.sign(d)
    return tail_pos
    
def tail_trail(num_knots, moves):
    knot_pos = [[0,0] for i in range(num_knots)]
    trail = [[0,0]]
    for line in q9input:
        move = line.split(' ')
        for x in range(int(move[1])):
            knot_pos[0] = move_head(knot_pos[0], move[0])
            for i in range(1, len(knot_pos)):
                knot_pos[i] = move_tail(knot_pos[i-1], knot_pos[i])
            trail.append(knot_pos[i].copy())
    return trail
    
#%% question 1
q1input = read_txt(r'C:\Users\Lenny\Documents\GitHub\AdventOfCode\2022\q1input.txt')
a = 0
b = [0,0,0]
for x in q1input:
    n = np.sum([int(y) for y in x])
    a = np.max((a, n))
    for m in range(3):
        if n >= b[m]:
            b.insert(m, n)
            b.pop(3)
            break
print(a, np.sum(b))

#%% question 2
q2input = read_txt(r'C:\Users\Lenny\Documents\GitHub\AdventOfCode\2022\q2input.txt')[0]
a = 0
b = 0
for x in q2input:
    them = get_score(x[0])
    you = get_score(x[2])
    outcome = resolve_game(them, you)
    a += you + outcome
    
    outcome2 = get_score2(x[2])
    you2 = invert_game(them, outcome2)
    b += you2 + outcome2
print(a, b)

#%% question 3    
q3input = read_txt(r'C:\Users\Lenny\Documents\GitHub\AdventOfCode\2022\q3input.txt')[0]
a = 0
b = 0
for x, i in zip(q3input, range(len(q3input))):
    s1 = x[:int(len(x)/2)]
    s2 = x[int(len(x)/2):]
    a += get_priority(matching_letter(s1,s2)[0])
    if np.mod(i,3) == 2:
        s1 = q3input[i]
        s2 = q3input[i-1]
        s3 = q3input[i-2]
        b += get_priority(matching_letter(matching_letter(s1,s2),s3)[0])
print(a, b)

#%% question 4
q4input = read_txt(r'C:\Users\Lenny\Documents\GitHub\AdventOfCode\2022\q4input.txt')[0]
a = 0
b = 0
for x in q4input:
    vals = [[int(n) for n in s.split('-')] for s in x.split(',')]
    s1 = set(range(vals[0][0], vals[0][1] + 1))
    s2 = set(range(vals[1][0], vals[1][1] + 1))
    a += (s1.issubset(s2) | s2.issubset(s1))
    b += (len(s1 & s2) > 0)
print(a, b)

#%% question 5
q5input = read_txt(r'C:\Users\Lenny\Documents\GitHub\AdventOfCode\2022\q5input.txt')

crates = q5input[0]
moves = q5input[1]
            
stacks = parse_crates(crates)
move_list = parse_moves(moves)

for m in move_list:
    num = m[0]
    frm = m[1]-1
    to = m[2]-1
    for n in range(num):
        stacks[to].append(stacks[frm].pop(-1))
print(''.join([x[-1] for x in stacks]))

stacks = parse_crates(crates)
move_list = parse_moves(moves)
for m in move_list:
    num = m[0]
    frm = m[1]-1
    to = m[2]-1
    temp = []
    for n in range(num):
        temp.append(stacks[frm].pop(-1))
    temp.reverse()
    stacks[to] = stacks[to]+temp
print(''.join([x[-1] for x in stacks]))

#%% question 6
q6input = read_txt(r'C:\Users\Lenny\Documents\GitHub\AdventOfCode\2022\q6input.txt')[0][0]

class signal_processor:
    
    def __init__(self, signal):
        self.signal = signal
    
    def check_for_packet(signal, idx, packet_length):
        subsignal = ''
        for i in range(packet_length):
            s = signal[idx + i]
            if s in subsignal:
                return False, None, None
            else:
                subsignal += s
        return True, subsignal, idx+i+1
    
    def get_packet_idx(self, packet_length):
        for idx in range(len(self.signal)-packet_length):
            has_packet, packet, packet_idx = self.check_for_packet(self.signal, idx, packet_length)
            if has_packet:
                return packet, packet_idx
        return None, None
    
        
print(signal_processor(q6input).get_packet_idx(4))
print(signal_processor(q6input).get_packet_idx(14))
    
#%%
q7test = read_txt(r'C:\Users\Lenny\Documents\GitHub\AdventOfCode\2022\q7test.txt')[0]
q7input = read_txt(r'C:\Users\Lenny\Documents\GitHub\AdventOfCode\2022\q7input.txt')[0]

class explorer:
    
    def __init__(self, codes):
        self.codes = codes
        self.dir = []
        self.idx = 0
    
    def change_dir(self, s):
        if s == '..':
            self.dir.pop(-1)
        else:
            self.dir.append(s)
        self.idx += 1
    
    def list_dir(self):
        dir_contents = []
        self.idx += 1
        while self.idx < len(self.codes):
            line = self.codes[self.idx].split(' ')
            if line[0] == '$':
                break
            elif line[0] == 'dir':
                pass
            else:
                for i in range(len(self.dir)):
                    dir_contents.append(['file', self.dir[:(i+1)], [line[1], line[0]]])
            self.idx += 1
        return dir_contents
    
    def process_command(self, line):
        if line[1] == 'cd':
            self.change_dir(line[2])
            return False, None
        elif line[1] == 'ls':
            dir_contents = self.list_dir()
            return True, dir_contents
        else:
            raise Exception('bad parsing1')
        
    def read_line(self):
        line = self.codes[self.idx].split(' ')
        if line[0] == '$':
            has_output, output = self.process_command(line)
        else:
            raise Exception('bad parsing2')
        return has_output, output
            
    def scan_directories(self):
        results = []
        while self.idx < len(self.codes):
            has_output, output = self.read_line()
            if has_output:
                results.extend(output)
        return results
            

contents = explorer(q7input).scan_directories()
df = pd.DataFrame(contents, columns = ['type', 'dir_list', 'file'])
df['dir'] = df['dir_list'].apply(lambda x: ''.join(x))
df['size'] = df['file'].apply(lambda x: int(x[1]))
sums = df.groupby('dir')['size'].sum()
print(np.sum(sums[sums <= 100000]))

max_size = 70000000
min_free_space = 30000000
free_space = max_size - sums.max()
cands = []
print(np.min(sums[(sums + free_space) > min_free_space]))

#%%
q8input = read_txt(r'C:\Users\Lenny\Documents\GitHub\AdventOfCode\2022\q8input.txt')[0]
q8test = read_txt(r'C:\Users\Lenny\Documents\GitHub\AdventOfCode\2022\q8test.txt')[0]
        
z = q8input.copy()
x = 0
y = []
for i in range(len(z)):
    for j in range(len(z[0])):
        o1, o2 = multidir_max(z, i, j)
        x += o1
        y.append(o2)
        
print(x, max(y))
    
#%% Question 9
q9input = read_txt(r'C:\Users\Lenny\Documents\GitHub\AdventOfCode\2022\q9input.txt')[0]
    
out = []
for t in tail_trail(2, q9input):
    if t in out:
        continue
    else:
        out.append(t)
print(len(out))

out = []
for t in tail_trail(10, q9input):
    if t in out:
        continue
    else:
        out.append(t)
print(len(out))

    
#%%
q10input = read_txt(r'C:\Users\Lenny\Documents\GitHub\AdventOfCode\2022\q10input.txt')[0]

c = 0
i = 0
x = 1
cd = 0
out = []
while True:
    
    if cd == 0:
        if i == len(q10input):
            break
        command = q10input[i]
        inst = command.split(' ')[0]
        i += 1
        if inst == 'noop':
            cd = 1
        elif inst == 'addx':
            val = int(command.split(' ')[1])
            cd = 2
    
    c += 1
    out.append([c, x])
    
    cd += -1
    if cd == 0:
        if inst == 'addx':
            x += val
            

    
z = 0    
for c in [20, 60, 100, 140, 180, 220]:
    print(out[c-1])
    z += c*out[c-1][1]
    
print(z) 


im = [['.' for i in range(40)] for i in range(6)]
for row in range(len(im)):
    for col in range(len(im[0])):
        idx = len(im[0])*(row) + col
        print(out[idx], out[idx][0] - row*len(im[0]))
        if (out[idx][0] - row*len(im[0]) - 1) in range(out[idx][1]-1, out[idx][1]+2):
            im[row][col] = '#'
im2 = []
for row2 in im:
    im2.append(''.join(row2))
print(im2)

#%%

q11input = read_txt(r'C:\Users\Lenny\Documents\GitHub\AdventOfCode\2022\q11input.txt')
q11test = read_txt(r'C:\Users\Lenny\Documents\GitHub\AdventOfCode\2022\q11test.txt')

class monkey:
    
    def __init__(self, num, items, op, test, throws, divide=True, gcf=1):
        self.num = num
        self.items = items
        self.op = op
        self.test = test
        self.throws = throws
        self.outs = []
        self.num_inspections = 0
        self.divide = divide
        self.gcf = gcf
        
    def inspect_items(self):
        for i in range(len(self.items)):
            old = self.items.pop(0)
            new = eval(self.op)
            if self.divide:
                new = int(np.floor(new/3))
            if self.gcf != 1:
                new = new % self.gcf
            self.num_inspections += 1
            if np.mod(new, self.test) == 0:
                self.outs.append([self.throws[0], new])
            else:
                self.outs.append([self.throws[1], new])

gcf = 1
for text in q11input:
    gcf = gcf*int(text[3].split(' ')[-1])
    
ops = []
monkey_list = []
for text in q11input:
    monkey_num = int(text[0].split(' ')[1][0])
    monkey_items = [int(x) for x in text[1].split(': ')[1].split(', ')]
    monkey_op = text[2].split('= ')[1]
    monkey_test = int(text[3].split(' ')[-1])
    ops.append(monkey_op)
    monkey_throws = [int(t.split(' ')[-1]) for t in text[4:]]
    monkey_list.append(monkey(monkey_num, monkey_items, monkey_op, monkey_test, monkey_throws, divide=True, gcf=gcf))
    
for n in range(20):
    out = []
    for x in monkey_list:
        x.inspect_items()
        for i in range(len(x.outs)):
            t = x.outs.pop(0)
            monkey_list[t[0]].items.append(t[1])
    
a = [0, -1]
for x in monkey_list:
    if x.num_inspections > min(a):
        a.remove(min(a))
        a.append(x.num_inspections)
print(a[0]*a[1])

ops=[]
monkey_list = []
for text in q11input:
    monkey_num = int(text[0].split(' ')[1][0])
    monkey_items = [int(x) for x in text[1].split(': ')[1].split(', ')]
    monkey_op = text[2].split('= ')[1]
    monkey_test = int(text[3].split(' ')[-1])
    ops.append(monkey_op)
    monkey_throws = [int(t.split(' ')[-1]) for t in text[4:]]
    monkey_list.append(monkey(monkey_num, monkey_items, monkey_op, monkey_test, monkey_throws, divide=False, gcf=gcf))
    
for n in range(10000):
    out = []
    for x in monkey_list:
        x.inspect_items()
        for i in range(len(x.outs)):
            t = x.outs.pop(0)
            monkey_list[t[0]].items.append(t[1])
    
a = [0, -1]
for x in monkey_list:
    if x.num_inspections > min(a):
        a.remove(min(a))
        a.append(x.num_inspections)
print(a[0]*a[1])

#%%
q12input = read_txt(r'C:\Users\Lenny\Documents\GitHub\AdventOfCode\2022\q12input.txt')[0]
q12test = read_txt(r'C:\Users\Lenny\Documents\GitHub\AdventOfCode\2022\q12test.txt')[0]

map = []
for row in q12input:
    map.append(['start' if v=='S' else 'end' if v == 'E' else ord(v)-96 for v in row])
for row, i in zip(map, range(len(map))):
    for y, j in zip(row, range(len(row))):
        if y == 'start':
            start = [i,j]
            map[i][j] = 1
        elif y=='end':
            end = [i,j]
            map[i][j] = 26

def check_incline(i,j,dx,dy):
    if (map[i+dx, j+dy] - map[i, j]) >= -1:
        return True
    else:
        return False

map = np.array(map)
dist = map + np.inf
visited = np.zeros(map.shape)
dist[end[0], end[1]] = 0
l = 1
min_val = 0

while True:
    min_val = np.min(dist[visited==0])
    ii, jj = np.where((visited==0) & (dist == min_val))
    if np.isinf(min_val) | (np.sum(visited==0) == 0):
        break
    for i, j in zip(ii, jj):
        visited[i,j] = 1
        for dx, dy in zip([0, 0, -1, 1], [-1, 1, 0, 0]):
            if (0 <= (i + dx) < map.shape[0]) & (0 <= (j + dy) < map.shape[1]):
                if check_incline(i,j,dx,dy):
                    dist[i+dx, j+dy] = np.min([min_val + 1, dist[i+dx, j+dy]])

print(dist[start[0]][start[1]])
print(np.min(dist[map==1]))

#%%
q13input = read_txt(r'C:\Users\Lenny\Documents\GitHub\AdventOfCode\2022\q13input.txt')
q13test = read_txt(r'C:\Users\Lenny\Documents\GitHub\AdventOfCode\2022\q13test.txt')

def compare_elements(l1, l2, i):
    if (type(l1) == list) & (type(l2) == list) & ((len(l1) == i) | (len(l2) == i)):
        if len(l1) < len(l2):
            return True, True
        elif len(l1) > len(l2):
            return True, False
        else:
            return False, None
    else:
        v1 = l1[i]
        v2 = l2[i]
        if (type(v1) == int) & (type(v2) == int):
            if v1 < v2:
                return True, True
            elif v1 > v2:
                return True, False
            else:
                return compare_elements(l1, l2, i+1)
        elif (type(v1) == list) & (type(v2) == list):
            out1, out2 = compare_elements(v1, v2, 0)
            if out1:
                return True, out2
            else:
                return compare_elements(l1, l2, i+1)
        elif (type(v1) == int) & (type(v2) == list):
            out1, out2 = compare_elements([v1], v2, 0)
            if out1:
                return True, out2
            else:
                return compare_elements(l1, l2, i+1)
        elif (type(v1) == list) & (type(v2) == int):
            out1, out2 = compare_elements(v1, [v2], 0)
            if out1:
                return True, out2
            else:
                return compare_elements(l1, l2, i+1)

r = []
for p, n in zip(q13test, range(1, 1+len(q13test))):
      s1 = eval(p[0])
      s2 = eval(p[1])
      if compare_elements(s1, s2, 0)[1]:
        r.append(n)

r = []
for p, n in zip(q13input, range(1, 1+len(q13input))):
      s1 = eval(p[0])
      s2 = eval(p[1])
      if compare_elements(s1, s2, 0)[1]:
        r.append(n)

print(sum(r))


full = []
for p in q13input:
    for q in p:
        full.append(eval(q))
full += [[[2]], [[6]]]

for m in range(len(full)):
    for n in range(len(full)-1):
        s1 = full[n]
        s2 = full[n+1]
        out1, out2 = compare_elements(s1, s2, 0)
        if not out2:
            full[n] = s2
            full[n+1] = s1
p = np.where([x == [[2]] for x in full])[0] + 1
q = np.where([x == [[6]] for x in full])[0] + 1
print(p*q)

#%%

#%%
q15input = read_txt(r'C:\Users\Lenny\Documents\GitHub\AdventOfCode\2022\q15input.txt')[0]
q15test = read_txt(r'C:\Users\Lenny\Documents\GitHub\AdventOfCode\2022\q15test.txt')[0]

sensors = []
beacons = []
for s in q15input:
    for t, i in zip(s.split(':'), range(2)):
        w = [int(x.split('=')[-1]) for x in t.split(', ')]
        if i == 0:
            sensors.append(w)
        else:
            beacons.append(w)
dist = np.sum(np.abs(np.array(sensors) - np.array(beacons)), axis=1)

sensors = np.array(sensors)

def get_x_for_y(yt, x1=-np.inf, x2=np.inf):
    xranges = []
    for xy, d in zip(sensors, dist):
        dx = d - np.abs(xy[1] - yt)
        if dx > 0:
            xmin = xy[0] - dx
            xmax = xy[0] + dx
            xranges.append([xmin, xmax])
    for i in range(len(xranges)):
        xranges[i][0] = np.max([x1, xranges[i][0]])
        xranges[i][1] = np.min([x2, xranges[i][1]])
    pop = True
    while pop:
        pop = False
        for i in range(len(xranges)):
            for j in range(len(xranges)):
                if i != j:
                    if xranges[i] == xranges[j]:
                        xranges.pop(j)
                        pop = True
                        break
                    elif xranges[i][0] <= xranges[j][0] <= xranges[i][1]:
                        xranges[i][0] = np.min([xranges[i][0], xranges[j][0]])
                        xranges[i][1] = np.max([xranges[i][1], xranges[j][1]])
                        xranges.pop(j)
                        pop = True
                        break
                    elif xranges[i][0] <= xranges[j][1] <= xranges[i][1]:
                        xranges[i][0] = np.min([xranges[i][0], xranges[j][0]])
                        xranges[i][1] = np.max([xranges[i][1], xranges[j][1]])
                        xranges.pop(j)
                        pop = True
                        break
            if pop:
                break
    a = 0
    for xr in xranges:
        a += xr[1] - xr[0]
    return a, xranges

print(get_x_for_y(2000000))

for yt in range(2671045-10, 2671045+10):
    x1 = 0
    x2 = 4000000
    o1, o2 = get_x_for_y(yt, x1=x1, x2=x2)
    if o1 < (x2-x1):
        print(yt, o1, o2)
        
#%%
# q16input = read_txt(r'C:\Users\Lenny\Documents\GitHub\AdventOfCode\2022\q16input.txt')[0]
# q16test = read_txt(r'C:\Users\Lenny\Documents\GitHub\AdventOfCode\2022\q16test.txt')[0]

# value = {}
# tunnel = {}
# for s in q16input:
#     v1 = s.split(' has flow')[0].split(' ')[-1]
#     v2 = int(s.split(' has flow')[1].split(';')[0].split('=')[-1])
#     if s.split('; tunnel')[1][0] == 's':
#         v3 = s.split(' valves ')[-1].split(', ')
#     else:
#         v3 = s.split(' valve ')[-1].split(', ')
#     value[v1] = v2
#     tunnel[v1] = v3
   
# def shortest_dist(x1_list, x2, n=1, bad_list=[]):
#     vals = []
#     for x1 in x1_list:
#         if x2 in tunnel[x1]:
#             return n
#         elif x1 not in bad_list:
#             vals += tunnel[x1]
#             new_bad_list = bad_list.copy()
#             new_bad_list.append(x1)
#     return shortest_dist(vals, x2, n=n+1, bad_list=new_bad_list)
    
# tunnel2 = {}
# for x1 in tunnel.keys():
#     vals = {}
#     for x2 in tunnel.keys():
#         if (x1 != x2) & (value[x2] > 0):
#             vals[x2] = shortest_dist([x1], x2)
#     tunnel2[x1] = vals
    
# def remove_destination(tunnels=[], dest=''):
#     for t1, t2 in tunnels.items():
#         if dest in t2.keys():
#             t2.pop(dest)
#             tunnels[t1] = t2
#     return tunnels

def relieve_pressure(pos='', time=0, flows={}, flow=0, total=0, path=[], tunnels=[]):
    
    # ran out of time, return the result
    if time == 1:
        # update the total based on the current flow
        new_total = total + flow
        # return output
        path.append([time, 'done', pos, '--', flows, flow, new_total])
        return int(new_total), path
    
    # no more moves left to make
    elif not any([x > 0 for x in flows.values()]):
        # update the total based on the current flow
        new_total = total + flow
        # continue waiting
        path.append([time, 'wait', pos, '--', flows, flow, new_total])
        return relieve_pressure(pos=pos, time=time-1, flows=flows, flow=flow, total=new_total, path=path, tunnels=tunnels)
    
    # make some moves
    else:
        
        # initialize output
        max_flow = []
        max_path = []
        
        # relieve pressure at current site if it isn't already open
        if flows[pos] > 0:
            # update the total based on the current flow
            new_total = total + flow
            # update the flow
            new_flows = flows.copy()
            new_flow = flow + new_flows[pos]
            # update the path
            new_path = path.copy()
            new_path.append([time, 'open', pos, new_flows[pos], new_flows, new_flow, new_total])
            # set the current site to zero
            new_flows[pos] = 0
            # remove the current site as a destination in the tunnel network
            new_tunnels = copy.deepcopy(tunnels)
            new_tunnels = remove_destination(tunnels=new_tunnels, dest=pos)
            # run this iteration
            out_total, out_path = relieve_pressure(pos=pos, time=time-1, flows=new_flows, flow=new_flow, total=new_total, path=new_path, tunnels=new_tunnels)
            max_flow.append(out_total)
            max_path.append(out_path.copy())
        
        # move to another tunnel
        for new_pos, steps_away in tunnels[pos].items():
            # only move if the valve is closed and there's enough time
            if (flows[new_pos] > 0) & ((time - steps_away) > 1):
                # update the total based on the current flow
                new_total = total + flow*steps_away
                # update the path
                new_path = path.copy()
                new_path.append([time, 'move', pos, new_pos, flows, flow, new_total])
                # remove the current site as a destination in the tunnel network
                new_tunnels = copy.deepcopy(tunnels)
                new_tunnels = remove_destination(tunnels=new_tunnels, dest=pos)
                new_tunnels = remove_destination(tunnels=new_tunnels, dest=new_pos)
                # run this iteration
                out_total, out_path = relieve_pressure(pos=new_pos, time=time-steps_away, flows=flows, flow=flow, total=new_total, path=new_path, tunnels=new_tunnels)
                max_flow.append(out_total)
                max_path.append(out_path.copy())
        
        # if no other options available then wait here
        if len(max_flow) == 0:
            # update the total based on the current flow
            new_total = total + flow
            # wait here
            path.append([time, 'wait', pos, '--', flows, flow, new_total])
            out_total, out_path = relieve_pressure(pos=pos, time=time-1, flows=flows, flow=flow, total=new_total, path=path, tunnels=tunnels)
            max_flow.append(out_total)
            max_path.append(out_path.copy())
        
        # return best result
        idx = np.where(np.array(max_flow) == max(max_flow))[0][0]
        return max_flow[idx], max_path[idx]

i1 = copy.deepcopy(value)
i2 = copy.deepcopy(tunnel2)
o1, o2 = relieve_pressure(pos='AA', time=30, flows=i1, flow=0, total=0, tunnels=i2)
for x in o2:
    x.pop(4)
    print(x)
print(o1)

#%%



#%% q18
q18input = read_txt(r'C:\Users\Lenny\Documents\GitHub\AdventOfCode\2022\q18input.txt')[0]
q18test = read_txt(r'C:\Users\Lenny\Documents\GitHub\AdventOfCode\2022\q18test.txt')[0]


xyz = []
for v in q18input:
    xyz.append(list(eval(v)))
    

# num_neighbors = []
# for v1 in xyz:
#     n=0
#     for v2 in xyz:
#         if v1!=v2:
#             if np.sum(np.abs(np.array(v1) - np.array(v2))) == 1:
#                 n+=1
#     num_neighbors.append(n)
# area = 0
# for n in num_neighbors:
#     area += 6-n
# print(area)


minv = np.min(np.array(xyz), axis=0)
maxv = np.max(np.array(xyz), axis=0)+1      
xyz_test = 
    


