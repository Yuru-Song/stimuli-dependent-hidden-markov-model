function dG = grad_emiss_filter(eta, V, S, Y)
% INPUT:
%   eta: [num_state, num_out, num_time], time-heterogenous emission matrix
%   V: [num_state, num_time], single marginal probability
%   S: [num_feat, num_time], stimuli
%   Y: [1, num_time], output
% OUTPUT:
%   dG: [num_state, num_out, num_feat], gradient of ECLL to emission
%   filters

% extract dimensions
[num_state, num_out, num_time] = size(eta);
num_feat = size(S,1);

dG = zeros(num_state, num_out, num_feat);

for n = 1: num_state
    for i = 1: num_out
        for t = 1: num_time
            dG(n, i, :) = squeeze(dG(n, i, :)) + squeeze(((Y(1, t) == i) - eta(n,i,t))*V(n,t)*S(:, t));
            if isnan(dG(n, i, :))
            error('NaN in dG');
            end
        end
    end
end