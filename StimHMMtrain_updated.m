function [trans_filter, emiss_filter, ECLL, U, V] = StimHMMtrain_updated(X, Y, options)
% Temporary info:
%   this is a temporary version function, to modify array format of trials
%   into cell format, DON'T forget to delete older version after
%   finishment! 
% INPUT:
%   X: features, [1, num_seq],cell array, each cell: {[num_feat x num_time double]}, continuous value
%   Y: output, [1, num_seq],cell array, each cell: {[1 x num_time integer]} discrete value, ranges in 1 ~ num_out
%   ---------------------
%   options:
%       num_state:  number of hidden states, default: 4
%       max_iter:  maximum training iteration, default: 1000
%       tol:  tolerance for objective function, default: 1e-5
%       rate:  learning rate, default: 1e-3
%       verbose:  displaying option during training, default: 0
%       disp_step:  display step, default: 10
%       least_sq:  whether to apply least square for fitting, default: 0
%       F:  initialization of transmission filter, default: ~randn()
%       G:  initialization of emission filter, default: ~randn()
%       lambda_F:  factor of L2 regularization, default: .1
%       lambda_G:  default: .1
% OUTPUT:
%   trans_filter: [num_state, num_state, num_feat +1 ]
%   emiss_filter: [num_state, num_out, num_feat + 1]
%   ECLL: [1, num_iter], expected complete log-likelihood over iterations
% NOTE:
%   this function append ones(1,num_time) to the features array, so you
%   don't need to worry about the bias term
% Yuru Song, Sept-28-2019

% get sequence number
num_seq = numel(X);
num_feat = zeros(1,num_seq);
% make sure the dimension is correct for features
for i = 1: num_seq
%     X{i} = X{i}';
    num_feat(i) = size(X{i}, 1);
end
if ~isequal(num_feat, num_feat(randperm(length(num_feat))))
    error('dimensions in feature data not consistent');
else
    num_feat = num_feat(1);
end
% get output features
num_out = 0;
for i = 1: num_seq
    num_out = max([num_out, max(Y{i})]);
end
% examine NaN values in inputs
for i =1: num_seq
    if any(isnan(X{i})) 
        error('nan in feature data');
    end
    if any(isnan(Y{i}))
        error('nan in output data');
    end
end

% training parameters
if ~exist('options','var')
    options.filler = 0;
end
if ~isfield(options, 'num_state')
    options.num_state = 3;
end
if ~isfield(options, 'max_iter')
    options.max_iter = 20;
end
if ~isfield(options, 'tol')
    options.tol = 1e-10;
end
if ~isfield(options, 'verbose')
    options.verbose = 1;
end
if ~isfield(options, 'disp_step')
    options.disp_step = 1;
end
if ~isfield(options, 'lambda_F')
    options.lambda_F = .1;
end
if ~isfield(options, 'lambda_G')
    options.lambda_G = .1;
end
num_state = options.num_state;
max_iter = options.max_iter;

if ~isfield(options, 'F')
    options.F = randn(num_state, num_state, num_feat+1);
    options.F(:,end,:) = 0;
    %eventually will rename it as trans_filter
end
if ~isfield(options, 'G')
    options.G = randn(num_state, num_out, num_feat+1);
    options.G(:,end,:) = 0;
%     options.G(:,:,end) = 1;
end
if ~isfield(options, 'debug')
    options.debug = 0;
end

tol = options.tol;
verbose = options.verbose; 
disp_step = options.disp_step;
debug = options.debug;
F = options.F; % be aware! F has bias term
G =  options.G;% be aware! G has bias term
ECLL = zeros(1, max_iter); % expected complete log-likelihood
lambda_F = options.lambda_F;
lambda_G = options.lambda_G;
if verbose
fprintf('Training stimuli dependent hidden Markov model...\n');
end

% check dimensions
if verbose
    fprintf('StimHMM dimension checking:\n')
    fprintf('num_state: %d, num_feat: %d, num_seq: %d\n', num_state, num_feat, num_seq);
end

%% Baum-Welch 
for iter = 1: max_iter
%     expectation
    alpha = cell(1, num_seq);
    eta = cell(1, num_seq);
    f = cell(1, num_seq);
    b = cell(1, num_seq);
    s = cell(1, num_seq);
    U = cell(1, num_seq);
    V = cell(1, num_seq);
    hstate = cell(1, num_seq);
    for seq = 1: num_seq
        alpha{seq} = compute_trans(X{seq}, F);
        eta{seq} = compute_emiss(X{seq}, G); 
        [f{seq}, b{seq}, s{seq}] = forward_backward(Y{seq}, alpha{seq}, eta{seq}); 
        U{seq} = pair_margin_prob(f{seq}, b{seq}, s{seq}, alpha{seq}, eta{seq}, Y{seq}); 
        V{seq} = single_margin_prob(f{seq}, b{seq});  
        ECLL(iter) = ECLL(iter) - compute_ECLL(U{seq}, alpha{seq}, V{seq}, eta{seq}, Y{seq}); 
        hstate{seq} = viterbi(Y{seq}, alpha{seq}, eta{seq});
    end
    
%     maximization: quasi Newton method
    for i = 1: num_state
        o.display = 0;

        F_tmp = reshape(F(i,:,:),[1,num_state*(num_feat + 1)]);
        F_tmp = F_tmp';
        funObj = @(F_tmp)trans_filter_learn_multi(U,F_tmp, X, i, lambda_F);  
         
        tmp_F = minFunc(funObj,F_tmp,o);
        F(i,:,:) = reshape(tmp_F,[1,num_state, (num_feat + 1)]); 
        if any(isnan(F(i,:)))
            error(['NaN in F(i,:,:), iter = ',num2str(iter)]);
        end
        
        G_tmp = reshape(G(i,:,:),[1,num_out*(num_feat + 1)]);
        G_tmp = G_tmp';
        funObj = @(G)emiss_filter_learn_multi(V, Y, G_tmp , X, i, lambda_G);  
                                                                                                                              
        tmp_G = minFunc(funObj, G_tmp,o);
        G(i,:,:) = reshape(tmp_G,[1,num_out,(num_feat + 1)]); 
        if any(isnan(G(i,:)))
            error(['NaN in G(i,:), iter = ',num2str(iter)]);
        end
    end


% % maximization: fit multinomial logistic regression
%     [F, G] = filter_mnr_learn(X, Y, hstate, num_state, num_out);

    if verbose > 0

        if mod(iter, disp_step) == 0
%             disp('-------------------------------');
            disp(['iter: ', num2str(iter), ', ECLL: ',num2str(ECLL(iter))]);
            disp(['max(F) = ',num2str(max(max(max(F))))]);
        end
    end
    if iter > 10 && abs(ECLL(iter) - ECLL(iter - 1)) < tol
         break;
    end
end

if abs(ECLL(iter) - ECLL(iter - 1))>tol
    fprintf('Warning: no convergence with tol %.3e after %d iterations...\n', tol, iter);
end
if nargout > 1
    trans_filter = F;
    emiss_filter = G;
end
