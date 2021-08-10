clearvars -except input input1 input2
close all
clc
%%
% input1 = 1;
%%
sz = (size(input1, 2) - 1)/2;
nameList = unique([input1(:, 1:sz); input1(:, (sz + 2):end)], 'rows');
adjMat = false(size(nameList, 1));
%%
for i = 1:size(nameList, 1)
    idx = ismember(input1(:, 1:sz), nameList(i, :), 'rows');
    idx2 = ismember(nameList, input1(idx, (sz + 2):end), 'rows');
    adjMat(i, :) = idx2';
end
%%
for n = 1:size(adjMat, 2)
    for i = 1:size(adjMat, 2)
        idx1 = adjMat(i, :);
        idx2 = adjMat(:, i);
        adjMat(idx2, idx1) = true;
%         for j = 1:length(idx2)
%             adjMat(idx2(j), idx1) = 1;
%         end
    end
end
%%
symAdjMat = adjMat | adjMat';
fw_symAdjMat = FloydWarshall(double(symAdjMat));
idx1 = find(ismember(nameList, 'YOU', 'rows'));
idx2 = find(ismember(nameList, 'SAN', 'rows'));
idx1_orbs = find(adjMat(:, idx1));
idx2_orbs = find(adjMat(:, idx2));

%%
% function count = DFS(nameList, idx, currCount)
%     idx = find(ismember(nameList,
% end
%%
function dist = FloydWarshall(adjMat)

dist = inf(size(adjMat));
dist(adjMat(:) > 0) = adjMat(adjMat(:) > 0);
dist(eye(size(dist)) > 0) = 0;

for k = 1:length(dist)
    dist = min(dist, dist(:, k) + dist(k, :));
end

dist(isinf(dist(:))) = nan;
end