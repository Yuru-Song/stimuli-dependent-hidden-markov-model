function U = pair_margin_prob(f, b, s, alpha, eta, Y)
% INPUT:
%   f: [num_state, num_time + 1], forward probability 
%   b: [num_state, num_time + 1], backward probability
%   s: [1, num_time +1], scaling factor in forward backward algorithm
%   alpha: [num_state, num_state, num_time], time-heterogenous transition
%   matrix
%   eta: [num_state, num_out, num_time], time-heterogenous emission matrix
%   Y: [1, num_time], output
% OUTPUT:
%   U: [num_state, num_state, num_time], consecutive pair-wise marginal
%   probability

% extract dimensions
[num_state, num_out, num_time] = size(eta);

U = zeros(num_state, num_state, num_time);
for t = 1: num_time-1
%     vectorize this when training is slow
    for m = 1: num_state
        for n = 1: num_state
            U(n,m,t) = f(n, t+1)*alpha(n,m,t)*eta(m, Y(1,t+1), t+1)*b(m, t+2);%t or t+1? check later
            if isnan(U(n,m,t))
                error('NaN in U(n,m,t)');
            end
        end
    end
    U(:,:,t) = U(:,:,t)/sum(sum(U(:,:,t)));
end
