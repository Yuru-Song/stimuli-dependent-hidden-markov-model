function V = single_margin_prob(f, b)
% INPUT:
%   f:[num_state, num_time + 1], forward probability
%   b:[num_state, num_time + 1], backward probability
% OUTPUT:
%   V:[num_state, num_time], single marginal probability

V = f(:, 2:end).*b(:, 1: end-1);
if isnan(V)
    error('NaN in V');
end
