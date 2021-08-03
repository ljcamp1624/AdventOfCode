idx = 1;
outputArray = inputArray;
while 1
    code = outputArray(idx);
    if code == 1 || code == 2
        p1 = outputArray(idx + 1) + 1;
        p2 = outputArray(idx + 2) + 1;        
        p3 = outputArray(idx + 3) + 1;
        v1 = outputArray(p1);
        v2 = outputArray(p2);
        if code == 1
            v3 = v1 + v2;
        elseif code == 2
            v3 = v1*v2;
        end
        outputArray(p3) = v3;
        idx = idx + 4;
%     elseif code == 3
%         p1 = outputArray(idx + 1) + 1;
%         v1 = outputArray(p1);
%         outputArray(p1) = v1;
%         idx = idx + 2;
    elseif code == 99
        break;
    end
end
[inputArray; outputArray];