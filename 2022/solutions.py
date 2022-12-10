#%%
import numpy as np
import pandas as pd

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
