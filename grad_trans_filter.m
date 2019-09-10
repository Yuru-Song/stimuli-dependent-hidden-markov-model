function dF = grad_trans_filter(alpha, U, S)
% INPUT:
%   alpha: [num_state, num_state, num_time], time-heterogenous transition
%   matrix
%   U: [num_state, num_state, num_time], consecutive pair-wise marginal
%   probability
%   S: [num_feat, num_time], stimuli
% OUTPUT:
%   dF: [num_state, num_state], gradient of ECLL to transition filters

% extract dimensions
[num_feat, num_time] = size(S);
num_state = size(U, 1);

dF = zeros(num_state, num_state, num_feat);

% VECTORIZE below if training is slow
for n = 1: num_state
    for j = 1: num_state
        for t = 1: num_time
            dF(n, j, :) = squeeze(dF(n, j, :)) + squeeze((U(n,j,t) - sum(U(n,:,t))*alpha(n,j,t))*S(:, t));
            if isnan(dF(n, j, :))
            error('NaN in dF');
            end
        end
    end
end
