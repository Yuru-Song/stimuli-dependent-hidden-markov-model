function [val, grad] = emiss_filter_learn_multi(V, Y, G_tmp, S, n, lambda_G)
% as in the case with multiple sequences, val and grad are simply the
% summation over all of the vals and grads from each seq.
% INPUT:
%   V: [1, num_seq] cell array, [num_state, num_time] in each cell
%   Y: [1, num_seq] cell array, [1, num_time] in each cell
%   G_tmp: [1, num_out* (num_feat + 1)]
%   S: [1, num_seq] cell array, [num_feat, num_time] in each cell
%   n: integer, tells us which slice of G_{n,:,:} we are dealing with
%   lambda_G: scalar regularization term
% OUTPUT:
%   val: scalar, value of Eq.3
%   grad: [1, num_out* (num_feat + 1)]

% extract dimension
num_seq = numel(S);

[val, grad] = emiss_filter_learn(V{1}(n, :, :), Y{1}, G_tmp, S{1}, lambda_G);
for seq = 2: num_seq
    [val_tmp, grad_tmp] = emiss_filter_learn(V{seq}(n, :, :), Y{seq}, G_tmp, S{seq}, lambda_G);
    val = val + val_tmp;
    grad = grad + grad_tmp;
end