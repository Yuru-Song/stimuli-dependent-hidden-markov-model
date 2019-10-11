function [Y, emiss_filter, trans_filter, features] = generate(num_feat, num_state, num_out, num_time, options)
% INPUT:
%   num_feat: number of feature in stimuli
%   num_state: number of hidden state
%   num_out: number of output choices
%   num_time: time length
%   G: optional, [num_state, num_out, num_feat + 1], emission filter, default: random 
%   F: optional, [num_state, num_state, num_feat + 1], transition filter, default: random
%   S: optional, [num_feat, num_time], stimuli, default: ones
% OUTPUT:
%   Y: [1, num_time], output over time
%   emiss_filter: [num_state, num_out, num_feat], just in case it's not
%   provided by user
%   trans_filter: [num_state, num_state, num_feat], also just in case it's
%   not provided by user
%   features: [num_feat, num_time], again, in case not provided by user

% SETUP
rng(1);
if ~isfield(options, 'S')
    options.S = rand(num_feat, num_time); %testing
end
num_feat = num_feat + 1;
if ~exist('options','var')
    options.filler = 0;
end
if ~isfield(options, 'G')
    options.G = rand(num_state, num_out, num_feat);
end
if ~isfield(options, 'F')
    options.F = rand(num_state, num_state, num_feat);
end
% for i = 1: num_feat
%     options.F(:,:,i) = options.F(:,:,i) - diag(diag(squeeze(options.F(:,:,i))));
% end
alpha = compute_trans(options.S, options.F);
eta = compute_emiss(options.S, options.G);
states = zeros(1,num_time);
Y = zeros(1, num_time);
% create two random sequences, one for state changes, one for emission
statechange = rand(1,num_time);
randvals = rand(1, num_time);
% Assume that we start in state 1.
currentstate = 1;
for t = 1: num_time
    alphac = cumsum(alpha(:,:,t),2);
    etac = cumsum(eta(:,:,t),2);
    % calculate state transition
    stateVal = statechange(t);
    state = 1;
    for innerState = num_state-1:-1:1
        if stateVal > alphac(currentstate,innerState)
            state = innerState + 1;
            break;
        end
    end
    % calculate emission
    val = randvals(t);
    emit = 1;
    for inner = num_out-1:-1:1
        if val  > etac(state,inner)
            emit = inner + 1;
            break
        end
    end
    % add values and states to output
    Y(t) = emit;
    states(t) = state;
    currentstate = state;
end

if nargout > 1
    emiss_filter = options.G;
    trans_filter = options.F;
    features = options.S;
end