clear all
close all
clc

%%
outputList = [];

%%
inputString = '3,225,1,225,6,6,1100,1,238,225,104,0,1002,148,28,224,1001,224,-672,224,4,224,1002,223,8,223,101,3,224,224,1,224,223,223,1102,8,21,225,1102,13,10,225,1102,21,10,225,1102,6,14,225,1102,94,17,225,1,40,173,224,1001,224,-90,224,4,224,102,8,223,223,1001,224,4,224,1,224,223,223,2,35,44,224,101,-80,224,224,4,224,102,8,223,223,101,6,224,224,1,223,224,223,1101,26,94,224,101,-120,224,224,4,224,102,8,223,223,1001,224,7,224,1,224,223,223,1001,52,70,224,101,-87,224,224,4,224,1002,223,8,223,1001,224,2,224,1,223,224,223,1101,16,92,225,1101,59,24,225,102,83,48,224,101,-1162,224,224,4,224,102,8,223,223,101,4,224,224,1,223,224,223,1101,80,10,225,101,5,143,224,1001,224,-21,224,4,224,1002,223,8,223,1001,224,6,224,1,223,224,223,1102,94,67,224,101,-6298,224,224,4,224,102,8,223,223,1001,224,3,224,1,224,223,223,4,223,99,0,0,0,677,0,0,0,0,0,0,0,0,0,0,0,1105,0,99999,1105,227,247,1105,1,99999,1005,227,99999,1005,0,256,1105,1,99999,1106,227,99999,1106,0,265,1105,1,99999,1006,0,99999,1006,227,274,1105,1,99999,1105,1,280,1105,1,99999,1,225,225,225,1101,294,0,0,105,1,0,1105,1,99999,1106,0,300,1105,1,99999,1,225,225,225,1101,314,0,0,106,0,0,1105,1,99999,108,677,677,224,102,2,223,223,1005,224,329,101,1,223,223,1107,677,226,224,102,2,223,223,1006,224,344,101,1,223,223,1107,226,226,224,102,2,223,223,1006,224,359,101,1,223,223,1108,677,677,224,102,2,223,223,1005,224,374,101,1,223,223,8,677,226,224,1002,223,2,223,1005,224,389,101,1,223,223,108,226,677,224,1002,223,2,223,1006,224,404,1001,223,1,223,107,677,677,224,102,2,223,223,1006,224,419,101,1,223,223,1007,226,226,224,102,2,223,223,1005,224,434,101,1,223,223,1007,677,677,224,102,2,223,223,1005,224,449,1001,223,1,223,8,677,677,224,1002,223,2,223,1006,224,464,101,1,223,223,1108,677,226,224,1002,223,2,223,1005,224,479,101,1,223,223,7,677,226,224,1002,223,2,223,1005,224,494,101,1,223,223,1008,677,677,224,1002,223,2,223,1006,224,509,1001,223,1,223,1007,226,677,224,1002,223,2,223,1006,224,524,1001,223,1,223,107,226,226,224,1002,223,2,223,1006,224,539,1001,223,1,223,1107,226,677,224,102,2,223,223,1005,224,554,101,1,223,223,1108,226,677,224,102,2,223,223,1006,224,569,101,1,223,223,108,226,226,224,1002,223,2,223,1006,224,584,1001,223,1,223,7,226,226,224,1002,223,2,223,1006,224,599,101,1,223,223,8,226,677,224,102,2,223,223,1005,224,614,101,1,223,223,7,226,677,224,1002,223,2,223,1005,224,629,101,1,223,223,1008,226,677,224,1002,223,2,223,1006,224,644,101,1,223,223,107,226,677,224,1002,223,2,223,1005,224,659,1001,223,1,223,1008,226,226,224,1002,223,2,223,1006,224,674,1001,223,1,223,4,223,99,226';
% inputString = '3,21,1008,21,8,20,1005,20,22,107,8,21,20,1006,20,31,1106,0,36,98,0,0,1002,21,125,20,4,20,1105,1,46,104,999,1105,1,46,1101,1000,1,20,4,20,1105,1,46,98,99';
% inputString = '3,9,8,9,10,9,4,9,99,-1,8';
% inputString = '3,12,6,12,15,1,13,14,13,4,13,99,-1,0,1,9';
% inputString = '3,3,1105,-1,9,1101,0,0,12,4,12,99,1';

% inputString = '3,21,1008,21,8,20,1005,20,22,107,8,21,20,1006,20,31,1106,0,36,98,0,0,1002,21,125,20,4,20,1105,1,46,104,999,1105,1,46,1101,1000,1,20,4,20,1105,1,46,98,99';
commaIdx = strfind(inputString, ',')';
locIdx = [[1; commaIdx + 1], [commaIdx - 1; length(inputString)]];
for i = 1:size(locIdx, 1)
    if strcmp(inputString(locIdx(i, 1):locIdx(i, 2)), '-')
        inputArray(i) = nan;
    else
        inputArray(i) = str2double(inputString(locIdx(i, 1):locIdx(i, 2)));
    end
end

%%
outputArray = inputArray;
idx = 1;
flag = 0;
while flag ~= 2
    % interpret code
    code = num2str(outputArray(idx), '%05u');
    op = str2double(code(4:5));
    param1 = str2double(code(3));
    param2 = str2double(code(2));
    param3 = str2double(code(1));
    
    % run op
    flag = 0;
    while flag ~= 1
        if op == 1 || op == 2
            if param1 == 0
                p1 = outputArray(idx + 1) + 1;
                v1 = outputArray(p1);
            elseif param1 == 1
                v1 = outputArray(idx + 1);
            end
            if param2 == 0
                p2 = outputArray(idx + 2) + 1;
                v2 = outputArray(p2);
            elseif param2 == 1
                v2 = outputArray(idx + 2);
            end
            if param3 == 0
                p3 = outputArray(idx + 3) + 1;
            elseif param3 == 1
                error('lies!');
            end
            if op == 1
                v3 = v1 + v2;
            elseif op == 2
                v3 = v1*v2;
            end
            outputArray(p3) = v3;
            idx = idx + 4;
            flag = 1;
        elseif op == 3
            v1 = input('input = ');
            if param1 == 0
                p1 = outputArray(idx + 1) + 1;
            elseif param1 == 1
                error('lies');
            end
            outputArray(p1) = v1;
            idx = idx + 2;
            flag = 1;
        elseif op == 4
            if param1 == 0
                p1 = outputArray(idx + 1) + 1;
                v1 = outputArray(p1);
            elseif param1 == 1
                p1 = outputArray(idx + 1) + 1;
                v1 = outputArray(idx + 1);
            end
            outputArray(p1) = v1;
            outputList = [outputList, v1];
            idx = idx + 2;
            flag = 1;
        elseif op == 5
            if param1 == 0
                p1 = outputArray(idx + 1) + 1;
                v1 = outputArray(p1);
            elseif param1 == 1
                v1 = outputArray(idx + 1);
            end
            if v1 ~= 0
                if param2 == 0
                    p2 = outputArray(idx + 2) + 1;
                    v2 = outputArray(p2) + 1;
                elseif param2 == 1
                    v2 = outputArray(idx + 2) + 1;
                end
                idx = v2;
                flag = 1;
            else
                idx = idx + 3;
                flag = 1;
            end
        elseif op == 6
            if param1 == 0
                p1 = outputArray(idx + 1) + 1;
                v1 = outputArray(p1);
            elseif param1 == 1
                v1 = outputArray(idx + 1);
            end
            if v1 == 0
                if param2 == 0
                    p2 = outputArray(idx + 2) + 1;
                    v2 = outputArray(p2) + 1;
                elseif param2 == 1
                    v2 = outputArray(idx + 2) + 1;
                end
                idx = v2;
                flag = 1;
            else
                idx = idx + 3;
                flag = 1;
            end
        elseif op == 7
            if param1 == 0
                p1 = outputArray(idx + 1) + 1;
                v1 = outputArray(p1);
            elseif param1 == 1
                v1 = outputArray(idx + 1);
            end
            if param2 == 0
                p2 = outputArray(idx + 2) + 1;
                v2 = outputArray(p2);
            elseif param2 == 1
                v2 = outputArray(idx + 2);
            end
            if param3 == 0
                p3 = outputArray(idx + 3) + 1;
            elseif param2 == 1
                error('lies!');
            end
            if v1 < v2
                outputArray(p3) = 1;
            else
                outputArray(p3) = 0;
            end
            idx = idx + 4;
            flag = 1;
        elseif op == 8
            if param1 == 0
                p1 = outputArray(idx + 1) + 1;
                v1 = outputArray(p1);
            elseif param1 == 1
                v1 = outputArray(idx + 1);
            end
            if param2 == 0
                p2 = outputArray(idx + 2) + 1;
                v2 = outputArray(p2);
            elseif param2 == 1
                v2 = outputArray(idx + 2);
            end
            if param3 == 0
                p3 = outputArray(idx + 3) + 1;
            elseif param2 == 1
                error('lies!');
            end
            if v1 == v2
                outputArray(p3) = 1;
            else
                outputArray(p3) = 0;
            end
            idx = idx + 4;
            flag = 1;
        elseif op == 99
            flag = 2;
            break;
        else
            error('sommething is wrong');
        end
    end
end
disp(['output = ', num2str(outputList)]);