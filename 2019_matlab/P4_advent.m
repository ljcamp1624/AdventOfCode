clear all
close all
clc

%%
minPass = [2, 0, 6, 9, 3, 8];
maxPass = [6, 7, 9, 1, 2, 8];

goodPassList = [];
badPassList = [];
for a = minPass(1):maxPass(1)
    for b = a:9
        for c = b:9
            for d = c:9
                for e = d:9
                    for f = e:9
                        pass = [a, b, c, d, e, f];
                        if CheckPass(pass)
                            goodPassList = [goodPassList; sum((10.^(5:-1:0)).*pass)];
                        else
                            badPassList = [badPassList; sum((10.^(5:-1:0)).*pass)];
                        end
                    end
                end
            end
        end
    end
end
goodPassList(goodPassList > sum((10.^(5:-1:0)).*maxPass)) = [];
badPassList(badPassList > sum((10.^(5:-1:0)).*maxPass)) = [];    

%% Check password
function out = CheckPass(pass)
    out = 1;
    %% check if 6 digits
    if pass(1) == 0
        out = 0;
        return;
    end
    %% check double digits
    vec = [diff(pass) == 0, false];
    vec2 = imdilate([0, diff(vec) == 0] & vec, [1 1 1]);
    if sum(vec) == 0
        out = 0;
        return;
%     elseif sum(vec(~vec2)) == 0
%         out = 0;
%         return;
    end
    %% check no decrease
    flag = 1;
    for i = 1:(length(pass) - 1)
        if pass(i + 1) < pass(i)
            flag = 0;
            break;
        end
    end
    if flag == 0
        out = 0;
        return;
    end
end