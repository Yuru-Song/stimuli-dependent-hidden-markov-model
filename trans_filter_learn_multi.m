function [val, grad] = trans_filter_learn_multi(U, F_tmp, S, n, lambda_F)
% as in the case with multiple sequences, val and grad are simply the
% summation over all of the vals and grads from each seq.
% INPUT:
%   U: [1, num_seq], cell array, {[num_state, num_state, num_time]} array in each cell
%   F_tmp:[1, num_state* (num_feat + 1)], is F_{n,m}, m \in 1~num_state, in Eq.7
%   S: features, [1, num_seq],cell array, each cell: {[num_feat x num_time double]}
%   n: integer, tells us which slice of F_{n,:,:} we are dealing with
%   lambda_F: scalar, regularization factor
% OUTPUT:
%   val: scalar, value of Eq. 2, a.k.a. Eq. 7
%   grad:[num_state* (num_feat+1)], value of Eq. 8

% extract dimension
num_seq = numel(U);
[val, grad] = trans_filter_learn(U{1}(n, :, :), F_tmp, S{1}, n, lambda_F);
for seq = 2: num_seq
    [val_tmp, grad_tmp] = trans_filter_learn(U{seq}(n, :, :), F_tmp, S{seq},n, lambda_F);
    val = val + val_tmp;
    grad = grad + grad_tmp;
end