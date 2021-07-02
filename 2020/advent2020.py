# %%
import numpy as np
import pandas as pd

# %% question 1
x = pd.read_csv("C:/Users/Lenny/Documents/Advent2020/q1.csv").values.flatten();
x2 = x[:, np.newaxis] + np.transpose(x[:, np.newaxis]);
ix,iy = np.where(x2==2020);
num = x[ix[0]]*x[iy[0]];
display(num);

y = x[:, np.newaxis, np.newaxis];
y2 = y + np.swapaxes(y, 0, 1) + np.swapaxes(y, 0, 2);
ix,iy,iz = np.where(y2==2020);
num = x[ix[0]]*x[iy[0]]*x[iz[0]];
display(num);

# %% question 2
x = pd.read_csv("C:/Users/Lenny/Documents/Advent2020/q2.csv", header=None);
x2 = x[0].str.split(pat="-| |:", expand=True)
x2[0] = x2[0].astype(int);
x2[1] = x2[1].astype(int);
for i in range(0, x2.shape[0]):
    x2.loc[i, 'count'] = x2.loc[i, 4].count(x2.loc[i, 2]);
x2['valid'] = (x2['count'] >= x2[0]) & (x2['count'] <= x2[1])
display(x2['valid'].sum())

for i in range(0, x2.shape[0]):
    x2.loc[i, 'letter1'] = x2.loc[i, 4][x2.loc[i, 0] - 1];
    x2.loc[i, 'letter2'] = x2.loc[i, 4][x2.loc[i, 1] - 1];
x2['check1'] = x2['letter1'] == x2[2];
x2['check2'] = x2['letter2'] == x2[2];
x2['valid2'] = np.logical_xor(x2['check1'], x2['check2']);
display(x2['valid2'].sum())

# %% question 3
x = pd.read_csv("C:/Users/Lenny/Documents/Advent2020/q3.csv", header=None);
x2 = x[0].str.split("",expand=True);
map = x2.drop(columns=[0, 32]).values;
slope = np.array([3, 1]);

def check_slope(map, slope):
    ix = np.arange(map.shape[0]).astype(int);
    iy = np.mod(ix*slope[0]/slope[1], map.shape[1]);
    good = iy == iy.astype(int);
    ix = ix[good].astype(int);
    iy = iy[good].astype(int);
    points = np.ravel_multi_index((ix,iy), map.shape);
    trees = np.sum(map.flatten()[points]=='#');
    return trees.astype(float);

prod = np.prod(np.array(
    [check_slope(map, [1, 1]), 
     check_slope(map, [3, 1]), 
     check_slope(map, [5, 1]), 
     check_slope(map, [7, 1]), 
     check_slope(map, [1, 2])]));
display(prod);

#%% question 4
x = pd.read_csv("C:/Users/Lenny/Documents/Advent2020/q4.csv", header=None, skip_blank_lines=False);
y = pd.DataFrame(columns=('byr', 'iyr', 'eyr', 'hgt', 'hcl', 'ecl', 'pid', 'cid'));
idx = 0;
for i in range(0, x.shape[0]):
    t = x.iloc[i, :].str.split(pat=":| ", expand=True);
    if pd.isna(t[0])[0]:
        idx = idx+1;
    else:
        for j in range(0, t.shape[1], 2):
            y.loc[idx, t[j].values] = t[j+1].values;
good = pd.isna(y.loc[:, (y.columns != 'cid')]).sum(axis=1) == 0;
good = good & (y['byr'].astype('float') >= 1920) & (y['byr'].astype('float') <= 2002);
good = good & (y['iyr'].astype('float') >= 2010) & (y['iyr'].astype('float') <= 2020);
good = good & (y['eyr'].astype('float') >= 2020) & (y['eyr'].astype('float') <= 2030);
hgt_num = y['hgt'].str.extract('(\d+)').astype('float');
hgt_unit = y['hgt'].str.extract('(\D+)');
good_cm = ((hgt_unit == 'cm') & ((hgt_num >= 150) & (hgt_num <= 193)))[0];
good_in = ((hgt_unit == 'in') & ((hgt_num >= 59) & (hgt_num <= 76)))[0];
good = good & (good_cm | good_in);
hcl_symb = y['hcl'].str[0];
hcl_hex_len = y['hcl'].str.split('#', expand=True)[1].str.split('', expand=True).isin(('a', 'b', 'c', 'd', 'e', 'f', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0')).sum(axis=1);
good = good & ((hcl_symb == '#') & (hcl_hex_len == 6));
good = good & (y['ecl'].isin(('amb', 'blu', 'brn', 'gry', 'grn', 'hzl', 'oth')));
good = good & (y['pid'].str.extract('(\d+)')[0].str.len()==9);
display(good.sum())

#%% question 5
x = pd.read_csv("C:/Users/Lenny/Documents/Advent2020/q5.csv", header=None);
x2 = x[0].str.split('', expand=True).replace('B',1).replace('F', -1).replace('R', 1).replace('L', -1).iloc[:, 1:11];
x3 = np.array([32, 16, 8, 4, 2, 1, .5, 2, 1, .5]);
x4 = x2*x3;
f_b = x4.iloc[:, 0:7].sum(axis=1)+127/2;
r_l = x4.iloc[:, 7:].sum(axis=1)+7/2;
seat_id = f_b*8 + r_l;
display(seat_id.max());
row = np.arange(0, 128);
col = np.arange(0, 8);
all_id_list = row[:, np.newaxis]*8+col[np.newaxis, :];
all_id_list = all_id_list.flatten();
elig = (all_id_list > 8) & (all_id_list < 1016);
all_id_list = pd.DataFrame({'seat':all_id_list.flatten(), 'elig':elig});
seat_taken = pd.DataFrame({'seat':seat_id, 'taken': 1});
all_id_list = all_id_list.merge(seat_taken, how='left', on='seat');

#%% question 6
x = pd.read_csv("C:/Users/Lenny/Documents/Advent2020/q6.csv", header=None, skip_blank_lines=False);
y = [''];
idx = 0;
for i in range(0, x.shape[0]):
    t = x.iloc[i, :];
    if pd.isna(t[0]):
        idx = idx+1;
    else:
        if len(y) == idx:
            y = np.append(y, t.values);
        else:
            y[idx] = y[idx]+t.values;
y2 = pd.DataFrame(y)
y3 = y2[0].str.split('', expand=True);