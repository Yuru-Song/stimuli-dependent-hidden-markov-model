function [val, grad] = emiss_filter_learn(V, Y, G, S)
% computes the value of Eq. 3 and 9 in README.md
% INPUT:
%   V: [1, num_time], V_{n,t}, t \in 1~num_time
%   Y: [1, num_time], all the output
%   G: [1, num_out, num_feat], G_{n,j}, j \in 1~num_state
%   S: [num_feat, num_time], all the stimuli
% OUTPUT:
%   val: scalar, value of Eq.3
%   grad: [1, num_out, num_feat], gradient of Eq.3 to G_{n,j}.
