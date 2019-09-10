function [val, grad] = trans_filter_learn(U, F, S)
% serves as a handle function passed to minfunc.m
% INPUT:
%   U:[1, num_state, num_time], actually is U_{n,m,t}, m \in 1~num_state, t \in 1~num_state, in Eq. 7
%   F:[1, num_state, num_feat], is F_{n,m}, m \in 1~num_state, in Eq.7
%   S:[num_feat, num_time], all the stimuli over time
% OUTPUT:
%   val: scalar, value of Eq. 2, a.k.a. Eq. 7
%   grad:[num_state, num_feat], value of Eq. 8

% extract dimensions
[num_feat, num_time] = size(S);
num_state = size(U,1);
%% compute Eq 7
val = 0;
alpha = zeros(1,num_state, num_time);% this is in fact alpha(n,m,t) in Eq 2, size: [1, num_state, num_time]
for t = 1: num_time
    alpha(1, :, t) = exp(sum(F.*repmat(reshape(S(:,t),[1, 1, num_feat]),[1, num_state]),3));
    val = val + sum(U(1,:,t).*alpha(1,:,t));
end
%% compute Eq 8
grad = zeros(1, num_state, num_feat);
for j = 1: num_state
    for t = 1: num_time
        grad(1, j, :) = grad(1, j, :) + (U(1, j, t) - sum(U(1,:,t))*alpha(1,j,t))*S(:, t);
    end
end