clear all
close all
clc

%%
permList = perms(0:4);
for i = 1:size(permList, 1)
    out = IntCode([permList(i, 1), 0]);
    for j = 2:5
        out = IntCode([permList(i, j), out]);
    end
    outList(i) = out;
end

%%
function [outputList, idx] = IntCode(inputList)
inputString = '3,8,1001,8,10,8,105,1,0,0,21,38,63,88,97,118,199,280,361,442,99999,3,9,1002,9,3,9,101,2,9,9,1002,9,4,9,4,9,99,3,9,101,3,9,9,102,5,9,9,101,3,9,9,1002,9,3,9,101,3,9,9,4,9,99,3,9,1002,9,2,9,1001,9,3,9,102,3,9,9,101,2,9,9,1002,9,4,9,4,9,99,3,9,102,2,9,9,4,9,99,3,9,102,4,9,9,101,5,9,9,102,2,9,9,101,5,9,9,4,9,99,3,9,1002,9,2,9,4,9,3,9,101,1,9,9,4,9,3,9,102,2,9,9,4,9,3,9,101,1,9,9,4,9,3,9,101,2,9,9,4,9,3,9,1001,9,2,9,4,9,3,9,102,2,9,9,4,9,3,9,102,2,9,9,4,9,3,9,101,1,9,9,4,9,3,9,102,2,9,9,4,9,99,3,9,101,1,9,9,4,9,3,9,102,2,9,9,4,9,3,9,1001,9,2,9,4,9,3,9,1001,9,2,9,4,9,3,9,1001,9,1,9,4,9,3,9,1001,9,1,9,4,9,3,9,1001,9,2,9,4,9,3,9,102,2,9,9,4,9,3,9,1002,9,2,9,4,9,3,9,1001,9,1,9,4,9,99,3,9,1002,9,2,9,4,9,3,9,1002,9,2,9,4,9,3,9,102,2,9,9,4,9,3,9,1002,9,2,9,4,9,3,9,1001,9,1,9,4,9,3,9,102,2,9,9,4,9,3,9,102,2,9,9,4,9,3,9,101,1,9,9,4,9,3,9,102,2,9,9,4,9,3,9,102,2,9,9,4,9,99,3,9,102,2,9,9,4,9,3,9,101,1,9,9,4,9,3,9,1002,9,2,9,4,9,3,9,1002,9,2,9,4,9,3,9,102,2,9,9,4,9,3,9,1002,9,2,9,4,9,3,9,1001,9,2,9,4,9,3,9,101,2,9,9,4,9,3,9,1001,9,2,9,4,9,3,9,101,1,9,9,4,9,99,3,9,101,1,9,9,4,9,3,9,101,1,9,9,4,9,3,9,101,1,9,9,4,9,3,9,102,2,9,9,4,9,3,9,1001,9,1,9,4,9,3,9,1001,9,2,9,4,9,3,9,101,2,9,9,4,9,3,9,102,2,9,9,4,9,3,9,1001,9,1,9,4,9,3,9,1001,9,2,9,4,9,99';
outputList = [];
commaIdx = strfind(inputString, ',')';
locIdx = [[1; commaIdx + 1], [commaIdx - 1; length(inputString)]];
for i = 1:size(locIdx, 1)
    if strcmp(inputString(locIdx(i, 1):locIdx(i, 2)), '-')
        inputArray(i) = nan;
    else
        inputArray(i) = str2double(inputString(locIdx(i, 1):locIdx(i, 2)));
    end
end
outputArray = inputArray;
idx = 1;
inputIdx = 1;
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
            v1 = inputList(inputIdx);
            inputIdx = inputIdx + 1;
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
end