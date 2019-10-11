function [val, grad] = emiss_filter_learn(V, Y, G_tmp, S, lambda_G)
% computes the value of Eq. 3 and 9 in README.md, serves as a handle function passed to minfunc.m
% INPUT:
%   V: [1, num_time], V_{n,t}, t \in 1~num_time
%   Y: [1, num_time], all the output
%   G_tmp: [1, num_out* (num_feat +1)], G_{n,j}, j \in 1~num_state
%   S: [num_feat, num_time], all the stimuli
%   lambda_G: scalar, regularization factor
% OUTPUT:
%   val: scalar, value of Eq.3
%   grad: [1, num_out* (num_feat + 1)], gradient of Eq.3 to G_{n,j}.

% extract dimensions
[num_feat, num_time] = size(S);

S = [S; ones(1, num_time)];
num_out = numel(G_tmp)/(num_feat + 1);
% reshape the flattened G
G_tmp = reshape(G_tmp, [1, num_out, num_feat+1]);
%% compute Eq 3
% get eta(n, j, t) first, j \in 1~num_out, t \in 1~num_time
eta = zeros(1, num_out, num_time);
val = 0;
for t = 1 : num_time
    eta(1, :, t) = exp(sum(G_tmp(1,:,:).*repmat(reshape(S(:, t), [1, 1, num_feat + 1]), [1, num_out]), 3));
    eta(1, 1,t) = 1;
    eta(1, :, t) = eta(1, :, t)./sum(eta(1, :, t), 2);
    if any(isnan(eta(1,:,t)))
        error('NaN in eta(1,:,t)');
    end  
    val = val + V(1, t).* log(eta(1,Y(t),t));
end

val = -val + sum(sum(G_tmp.^2)) * lambda_G;
%% compute Eq 9
grad = zeros(1, num_out, num_feat + 1);
for j = 1: num_out-1
    for t = 1: num_time
        grad(1, j, :) = grad(1, j, :) + ((Y(1,t) == j)*(~(Y(1,t) == 1)) - eta(1, j, t)*(~(j==1)))*V(1, t)*reshape(S(:, t),[1,1,num_feat + 1]);
        if any(isnan(grad(1, j,:)))
            error('NaN in grad(1, i, :)');
        end 
    end
end
grad = reshape(grad,[1, num_out*(num_feat + 1)]);
if nargout>1
grad = -grad'+ 2* reshape(G_tmp,[num_out*(num_feat+1),1])* lambda_G;
end
