function [val, grad] = trans_filter_learn(U, F_tmp, S, n, lambda_F)
% computes the value of Eq. 2 and 8 in README.md, serves as a handle function passed to minfunc.m
% INPUT:
%   U:[1, num_state, num_time], actually is U_{n,m,t}, m \in 1~num_state, t \in 1~num_state, in Eq. 7
%   F_tmp:[1, num_state* (num_feat+1)], is F_{n,m}, m \in 1~num_state, in Eq.7
%   S:[num_feat, num_time], all the stimuli over time
%   n: integer, tells us which slice of F_{n,:,:} we are dealing with
%   lambda_F: scalar, regularization factor
% OUTPUT:
%   val: scalar, value of Eq. 2, a.k.a. Eq. 7
%   grad:[num_state* (num_feat+1)], value of Eq. 8

% extract dimensions
[num_feat, num_time] = size(S);
S = [S; ones(1, num_time)];
num_state = size(U,2);
% reshape the flattened F
F_tmp = reshape(F_tmp, [1, num_state, num_feat + 1]);
%% compute Eq 7
val = 0;
alpha = zeros(1,num_state, num_time);% this is in fact alpha(n,m,t) in Eq 2, size: [1, num_state, num_time]
for t = 1: num_time
    alpha(1, :, t) = exp(sum(F_tmp(1,:,:).*repmat(reshape(S(:,t),[1, 1, numel(S(:,t))]),[1, num_state]),3));
    alpha(1, :, t) = alpha(1, :, t) ./ sum(alpha(1, :, t), 2);
    if any(isnan(alpha(1,:,t)))
        error('NaN in alpha(1,:,t)');
    end    
    val = val + sum(U(1,:,t).*log(alpha(1,:,t)));
end
val = -val + sum(sum(F_tmp.^2)) * lambda_F;
%% compute Eq 8
grad = zeros(1, num_state, num_feat + 1);
for i = 1: num_state-1
    
    for t = 1: num_time-1
        grad(1, i, :) = grad(1, i, :) + (U(1, i, t) - sum(U(1,:,t))*alpha(1,i,t))*reshape(S(:, t),[1, 1, num_feat+1]);
        if any(isnan(grad(1, i, :)))
            error('NaN in grad(1, j, :)');
        end    
    end
end

% disp(grad);pause();
grad = reshape(grad, [1, num_state* (num_feat+1)]);

if nargout>1
grad = -grad' + 2* reshape(F_tmp,[num_state*(num_feat+1),1])* lambda_F;
end
