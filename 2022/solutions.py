#%%
import numpy as np

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

def parse_moves(moves):
    move_list = []
    for s in moves:
        move_list.append([int(x) for x in s.split(' ')[1::2]])
    return move_list
    
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
                dir_contents.append(['dir', self.dir.copy(), None])
            else:
                dir_contents.append(['file', self.dir.copy(), [line[1], line[0]]])
            self.idx += 1
    
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
        if has_output:
            return output
            
    def scan_directories(self, return_results = False):
        results = []
        while self.idx < len(self.codes):
            results.append(self.read_line(return_results = True))
        if return_results:
            return results
            
x = explorer(q7input)
print(x.scan_directories())