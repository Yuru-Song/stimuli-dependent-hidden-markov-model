function [F, G] = filter_mnr_learn(S, output, hstate, num_state, num_out)
% INPUT: 
%   S: [1, num_seq] cell array, each cell: {[num_feat x num_time double]}, continuous value
%   output: [1, num_seq] cell array, each cell: {[1 x num_time integer]}
%   hstate: [1, num_seq] cell array, each cell: {[1 x num_time integer]}
%   num_state: integer
%   num_out: integer
% OUTPUT:
%   F: [num_state, num_state, num_feat + 1]
%   G: [num_state, num_out, num_feat + 1]
% @Oct-04-2019, Yuru Song

num_seq = numel(S);
num_feat = size(S{1}, 1);
F = zeros(num_state, num_state, num_feat + 1);
G = zeros(num_state, num_out, num_feat + 1);
for i = 1: num_state
    % prepare for mrnfit
    X = [];
    Y1 = [];
    Y2 = [];
    for seq = 1: num_seq
        num_time = size(S{seq},2);
        for time = 1: num_time-1
            if hstate{seq}(time) == i
                X = [X; S{seq}(:, time)'];
                Y1 = [Y1; 1:num_state == hstate{seq}(time + 1)];
                Y2 = [Y2; 1:num_out == output{seq}(time + 1)];
            end
        end
    end
    F(i, 1:num_state-1, :)  = mnrfit(X, Y1,'Interaction','on')';
    G(i, 1:num_out-1, :) = mnrfit(X, Y2, 'Interaction','on')';
end
