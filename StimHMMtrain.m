function [trans_filter, emiss_filter, ECLL, U, V] = StimHMMtrain(features, output, options)
% INPUT:
%   features:[num_seq, num_feat, num_time], continuous value
%   output:[num_seq, 1, num_time], discrete value, ranges in 1 ~ num_out
%   num_state: optional, number of hidden states, default: 4
%   max_iter: optional, maximum training iteration, default: 1000
%   tol: optional, tolerance for objective function, default: 1e-5
%   rate: optional, learning rate, default: 1e-3
%   verbose: optional, displaying option during training, default: 0
%   disp_step: optional, display step, default: 10
%   least_sq: optional, whether to apply least square for fitting, default: 0
%   F: optional, initialization of transmission filter, default: ~randn()
%   G: optional, initialization of emission filter, default: ~randn()
%   lambda_F: optional, regularization factor, default: .1
%   lambda_G: optional, regularization factor, default: .1
% OUTPUT:
%   trans_filter: [num_state, num_state, num_feat]
%   emiss_filter: [num_state, num_out, num_feat]
%   ECLL: [1, num_iter], expected complete log-likelihood over iterations
% NOTE:
%   append ones(1,num_time) to the features array before feeding it to the
%   function, so that there is no annoying bias term showing up in this function
% 
% Yuru Song, Sept-08-2019
% 

% extract dimensitons
[num_seq, num_feat, num_time] = size(features);
num_out = max(max(output));
% training term


if ~exist('options','var')
    options.filler = 0;
end
if ~isfield(options, 'num_state')
    options.num_state = 2;
end
if ~isfield(options, 'max_iter')
    options.max_iter = 100;
end
if ~isfield(options, 'tol')
    options.tol = 1e-20;
end
if ~isfield(options, 'rate')
    options.rate = 1;
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
    options.F = rand(num_state, num_state, num_feat).*repmat(~eye(num_state),[1,1,num_feat]);
    %eventually will rename it as trans_filter
end
if ~isfield(options, 'G')
    options.G = rand(num_state, num_out, num_feat);
    options.G(:,1,:) = 0;
end
if ~isfield(options, 'debug')
    options.debug = 0;
end

% least_sq = options.least_sq;
% refer readme.md for notations
S = features;
Y = output;
tol = options.tol;
rate = options.rate;
verbose = options.verbose; 
disp_step = options.disp_step;
debug = options.debug;
lambda_F = options.lambda_F;
lambda_G = options.lambda_G;
%     stupid NaN screening
    if any(isnan(S))
        error(['NaN in S, iter = ',num2str(iter)]);
    end
    if any(isnan(Y))
        error(['NaN in Y, iter = ',num2str(iter)]);
    end
%     end screening
alpha = zeros(num_seq, num_state, num_state, num_time); %discard the last column in time
eta = zeros(num_seq, num_state, num_out, num_time);
F = options.F;
G =  options.G;%and rename this as emiss_filter
f = zeros(num_seq, num_state, num_time+1);
b = zeros(num_seq, num_state, num_time+1);
s = zeros(num_seq, 1, num_time+1);
U = zeros(num_seq, num_state, num_state, num_time); % same as p(q_t-1 = m, q_t = m), also discard the last column in time
V = zeros(num_seq, num_state, num_time); % same as p(q_t = m)
ECLL = zeros(1, max_iter); % expected complete log-likelihood
% start training
if verbose
fprintf('Training stimuli dependent hidden Markov model...\n');
end
for iter = 1: max_iter
%% expectation
%    get time-heterogeneous transition matrix alpha
    for seq = 1: num_seq
    alpha(seq,:,:,:) = compute_trans(reshape(S(seq,:,:),[num_feat, num_time]), F);
    end
    if any(isnan(alpha))
        error(['NaN in alpha, iter = ',num2str(iter)]);
    end
%    also get time-heterogeneous emission matrix eta
    for seq = 1: num_seq
    eta(seq,:,:,:) = compute_emiss(reshape(S(seq,:,:),[num_feat, num_time]), G);
    end
    if any(isnan(eta))
        error(['NaN in eta, iter = ',num2str(iter)]);
    end
%    forward-backward 
    for seq = 1: num_seq
    [f(seq,:,:), b(seq,:,:), s(seq,:,:)] = forward_backward(reshape(Y(seq,:,:),[1,num_time]),reshape(alpha(seq,:,:,:),[num_state, num_state, num_time]),reshape(eta(seq,:,:,:),[num_state, num_out, num_time]));
    end
    if sum(f(:, end)) == 0
        error('P(y|theta) = 0, i.e. sum(f(:, end))= 0 ');
    end
%     get pair wise consecutive marginal probabilty
    for seq = 1: num_seq
    U(seq, :,:,:) = pair_margin_prob(reshape(f(seq,:,:),[num_state, num_time+1]), reshape(b(seq,:,:),[num_state, num_time+1]), reshape(s(seq,:,:),[1, num_time+1]),reshape(alpha(seq,:,:,:),[num_state, num_state, num_time]), reshape(eta(seq,:,:,:),[num_state, num_out, num_time]), reshape(Y(seq,:,:),[1,num_time]));
    end
    if any(isnan(U))
        error(['NaN in U, iter = ',num2str(iter)]);
    end
%     get single marginal probabilty
    for seq = 1: num_seq
    V(seq, :,:) = single_margin_prob(reshape(f(seq,:,:),[num_state, num_time + 1]), reshape(b(seq,:,:),[num_state, num_time + 1]));
    end
    if any(isnan(V))
        error(['NaN in V, iter = ',num2str(iter)]);
    end
    for seq = 1: num_seq
    ECLL(iter) = ECLL(iter) - compute_ECLL(reshape(U(seq, :,:,:),[num_state, num_state, num_time]), reshape(alpha(seq,:,:,:),[num_state, num_state, num_time]), reshape(V(seq, :,:),[num_state, num_time]), reshape(eta(seq,:,:,:),[num_state, num_out, num_time]), reshape(Y(seq,:,:),[1,num_time]));
    end
    if any(isnan(ECLL))
        error(['NaN in ECLL, iter = ',num2str(iter)]);
    end
%% maximization

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % %  first I tried using steepest gradient ascent, which was STUPID
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % %   compute the gradient of transmission filter
% %     dF = grad_trans_filter(alpha, U, S);
% %     if any(isnan(dF))
% %         error(['NaN in dF, iter = ',num2str(iter)]);
% %     end
% % %   compute the gradient of emission filter
% %     dG = grad_emiss_filter(eta, V, S, Y);
% %     if any(isnan(dG))
% %         error(['NaN in dG, iter = ',num2str(iter)]);
% %     end
% % %    try direct gradient ascent, if bad performance, resort to minfunc
% % %    package instead
% %     F = F + rate * dF;
% %     if any(isnan(F))
% %         error(['NaN in F, iter = ',num2str(iter)]);
% %     end
% %     G = G + rate * dG;
% %     if any(isnan(G))
% %         error(['NaN in G, iter = ',num2str(iter)]);
% %     end  
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

% % then I seek help from the minfunc package, optimize slices of F and G
% % seperately, i.e. feed F(n, :, :) and G(n, :, :) into minfunc each time
% % BUT this still doesn't work, since minfunc doesn't take good care of
% % vector variables, shit...
% % now I have to manually flatten each F(i,:,:) and G(i,:,:)
% %     fprintf('Maximizing Eq. 2 & 3...\n');
% %     o.method = 'csd';
% %     if ~least_sq
    o.display = 0;
%     o.Method = 'csd';
    if debug> 0
        imagesc(squeeze(F));colorbar;pause();close;
    end
    for i =1 : num_state
        F_tmp = reshape(F(i,:,:),[1,num_state*num_feat]);
        F_tmp = F_tmp';
        
        funObj = @(F_tmp)trans_filter_learn_multi(U(:,i,:,:),F_tmp, S,i, lambda_F);
        
        tmp_F = minFunc(funObj,F_tmp,o);
        F(i,:,:) = reshape(tmp_F,[1,num_state,num_feat]);
        if any(isnan(F(i,:)))
            error(['NaN in F(i,:,:), iter = ',num2str(iter)]);
        end
    end
    for i = 1: num_state
        G_tmp = reshape(G(i,:,:),[1,num_out*num_feat]);
        G_tmp = G_tmp';
        funObj = @(G)emiss_filter_learn_multi(V(:,i,:), Y, G_tmp , S, lambda_G);
        tmp_G = minFunc(funObj, G_tmp,o);
        G(i,:,:) = reshape(tmp_G,[1,num_out,num_feat])*rate;
        if any(isnan(G(i,:)))
            error(['NaN in G(i,:), iter = ',num2str(iter)]);
        end
    end
%     else
% %         do least square here 
%     end

% % % % % % % % % % % % % % % % % % % % 
% % minfunc package fails so bad. I don't know if its because my
% % pair_margin_prob.m is wrong (?) or whatever
% % try treat it as multinomial logistic regression
% % now the generate.m is needed.


    if verbose > 0
        if mod(iter, disp_step) == 0
%             disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
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
