function  ECLL = compute_ECLL(U, alpha, V, eta, Y)
% INPUT:
%   U: [num_state, num_state, num_time], consecutive pair-wise marginal
%   alpha: [num_state, num_state, num_time], time-heterogenous transition
%   matrix
%   V: [num_state, num_time], single marginal probability
%   eta: [num_state, num_out, num_time], time-heterogenous emission matrix
%   Y: [1, num_time], output
% OUTPUT:
%   ECLL: scalar, expected complete log-likelihood

% extract dimensions
[num_state, num_out, num_time] = size(eta);

ECLL = 0;
for t = 1: num_time
    for n = 1: num_state
        for m = 1: num_state
            ECLL = ECLL + U(n,m,t)*log(alpha(n,m,t));
        end
    end
end
for t = 1: num_time
    for n = 1: num_state        
        ECLL = ECLL + V(n,t)*log(eta(n,Y(1,t),t));
    end
end
