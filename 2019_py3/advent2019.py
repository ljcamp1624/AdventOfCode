#%% Advent of code 2019
import numpy as np
import pandas as pd

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

x = np.array([1,12,2,3,1,1,2,3,1,3,4,3,1,5,0,3,2,1,9,19,1,5,19,23,1,6,23,27,1,27,10,31,1,31,5,35,2,10,35,39,1,9,39,43,1,43,5,47,1,47,6,51,2,51,6,55,1,13,55,59,2,6,59,63,1,63,5,67,2,10,67,71,1,9,71,75,1,75,13,79,1,10,79,83,2,83,13,87,1,87,6,91,1,5,91,95,2,95,9,99,1,5,99,103,1,103,6,107,2,107,13,111,1,111,10,115,2,10,115,119,1,9,119,123,1,123,9,127,1,13,127,131,2,10,131,135,1,135,5,139,1,2,139,143,1,143,5,0,99,2,0,14,0]);
print(run_intcode(x))

for n in range(100):
    for m in range(100):
        y = x.copy();
        y[1] = n;
        y[2] = m;
        if run_intcode(y) == 19690720:
            print(n,m,100*n+m);

            





