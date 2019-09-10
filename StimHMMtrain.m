function [trans_filter, emiss_filter, ECLL, Pytheta] = StimHMMtrain(features, output, options)
% INPUT:
%   features:[num_feat, num_time], continuous value
%   output:[1, num_time], discrete value, ranges in 1 ~ num_out
%   num_state: optional, number of hidden states, default: 4
%   max_iter: optional, maximum training iteration, default: 1000
%   tol: optional, tolerance for objective function, default: 1e-5
%   rate: optional, learning rate, default: 1e-3
%   verbose: optional, displaying option during training, default: 0
% OUTPUT:
%   trans_filter: [num_state, num_state, num_feat]
%   emiss_filter: [num_state, num_out, num_feat]
% NOTE:
%   append ones(1,num_time) to the features array before feeding it to the
%   function, so that there is no annoying bias term showing up in this function
% Yuru Song, Sept-08-2019

if ~exist('options','var')
    options.filler = 0;
end
if ~isfield(options, 'num_state')
    options.num_state = 4;
end
if ~isfield(options, 'max_iter')
    options.max_iter = 10000;
end
if ~isfield(options, 'tol')
    options.tol = 1e-5;
end
if ~isfield(options, 'rate')
    options.rate = 1e-3;
end
if ~isfield(options, 'verbose')
    options.verbose = 0;
end

% SETUPS
% extract dimensitons
[num_feat, num_time] = size(features);
num_out = max(output);
% training term
num_state = options.num_state;
max_iter = options.max_iter;
tol = options.tol;
rate = options.rate;
verbose = options.verbose; 
% refer readme.md for notations
S = features;
Y = output;
%     stupid NaN screening
    if any(isnan(S))
        error(['NaN in S, iter = ',num2str(iter)]);
    end
    if any(isnan(Y))
        error(['NaN in Y, iter = ',num2str(iter)]);
    end
%     end screening
% alpha = zeros(num_state, num_state, num_time); 
F = rand(num_state, num_state, num_feat);%eventually will rename it as trans_filter
% dF = zeros(num_state, num_state, num_feat); % later computed from \partial ECLL/\partial U in the maximization part
% eta = zeros(num_state, num_out, num_time);
G = rand(num_state, num_out, num_feat); %and rename this as emiss_filter
% dG = zeros(num_state, num_out, num_feat); % later computed from \partial ECLL/\partial V in the maximization part, also
% U = rand(num_state, num_state, num_time); % same as p(q_t-1 = m, q_t = m)
% V = zeros(num_state, num_time); % same as p(q_t = m)
ECLL = zeros(1, max_iter); % expected complete log-likelihood
logpSeq = zeros(1, max_iter);
% start training
for iter = 1: max_iter
%% expectation
%    get time-heterogeneous transition matrix alpha
    alpha = compute_trans(S, F);
    if any(isnan(alpha))
        error(['NaN in alpha, iter = ',num2str(iter)]);
    end
%    also get time-heterogeneous emission matrix eta
    eta = compute_emiss(S, G);
    if any(isnan(eta))
        error(['NaN in eta, iter = ',num2str(iter)]);
    end
%    forward-backward 
    [f, b, s] = forward_backward(Y,alpha,eta);
    if sum(f(:, end)) == 0
        error('P(y|theta) = 0, i.e. sum(f(:, end))= 0 ');
    end
%     get ECLL
    logpSeq(iter) = sum(log(s)); % this can quickly get the ECLL, 
%     but don't forget the consecutive pairwise mariginal probabilty, i.e. U
    if any(isnan(logpSeq))
        error(['NaN in logpSeq, iter = ',num2str(iter)]);
    end
    U = pair_margin_prob(f, b, alpha, eta, Y);
    if any(isnan(U))
        error(['NaN in U, iter = ',num2str(iter)]);
    end
%     also the adorable single marginal probability, i.e. V
    V = single_margin_prob(f, b);
    if any(isnan(V))
        error(['NaN in V, iter = ',num2str(iter)]);
    end
    ECLL(iter) = compute_ECLL(U, alpha, V, eta, Y);
    if any(isnan(ECLL))
        error(['NaN in ECLL, iter = ',num2str(iter)]);
    end
%% maximization
%   compute the gradient of transmission filter
    dF = grad_trans_filter(alpha, U, S);
    if any(isnan(dF))
        error(['NaN in dF, iter = ',num2str(iter)]);
    end
%   compute the gradient of emission filter
    dG = grad_emiss_filter(eta, V, S, Y);
    if any(isnan(dG))
        error(['NaN in dG, iter = ',num2str(iter)]);
    end
%    try direct gradient ascent, if bad performance, resort to minfunc
%    package instead
    F = F + rate * dF;
    if any(isnan(F))
        error(['NaN in F, iter = ',num2str(iter)]);
    end
    G = G + rate * dG;
    if any(isnan(G))
        error(['NaN in G, iter = ',num2str(iter)]);
    end
    if verbose > 0
        if mod(iter, 10000) == 0
            disp(['iter: ', num2str(iter), ', ECLL: ',num2str(ECLL(iter)), ', pSeqs: ',num2str(logpSeq(iter))]);
        end
    end  
%     if iter > 100 && ECLL(end) - ECLL(end - 100) < tol
%     break;
%     end
end
if nargout > 1
    trans_filter = F;
    emiss_filter = G;
    Pytheta = logpSeq;
end
