function [K1, H1, K2, H2, L, alpha, eta] = StimHistHMMtrain(Nstate, Ntau, Niter, S, Y, rate)
% StimHistHMMtrain trains the linear filters of a hidden Markov model with dependence on stimuli and history.
% INPUTS:
%   Nstate: integer, number of discrete hidden states 
%   Ntau: integer, length of history dependence 
%   Niter: integer, iteration times of expectation-maximization(EM) algorithm
%   S: [Nstim, Nt] array, continuous external stimuli, with Nstim features 
%   Y: [1, Nt] array, discrete output, varies in [1, 2, .., Nout], Nout is the number of types of output
%   rate: float, learning rate
% OUTPUTS:
%   K1: [Nstate, Nstate, Nstim] array, linear filter of stimuli in transition between hidden states
%   H1: [Nstate, Nstate, Ntau] array, linear filter of history in transiton transition between hidden states
%   K2: [Nstate, Nout, Nstim] array, linear filter of stimuli in emission
%   H2: [Nstate, Nout, Ntau] array, linear filter of history in emission
%   L: [1, Niter] array, expected complete log-likelihood of parameters
% reference: Escola et al, Neural Computation, 2010
% @Yuru Song, Sept-02-2019, UCSD

%extract dimensions
[Nstim, Nt] = size(S);
S_rescale = rescale(S);
Nout = max(Y);
%random initialization of filters
K1 = rand(Nstate, Nstate, Nstim);
H1 = rand(Nstate, Nstate, Ntau);
K2 = rand(Nstate, Nout, Nstim);
H2 = rand(Nstate, Nout, Ntau);
for i = 1: Nstate
    K1(i,i,:) = 0;
    H1(i,i) = 0;
end
%off set Y by Ntau for the convenience of initial steps of filtering
Y = [zeros(1, Ntau),Y];
Y_rescale = rescale(Y);
%objective function
L = zeros(1,Niter);
for iter = 1: Niter
    %% expectation    
    % get alpha, the real transition matrix, which is time-inhomogenous
    % also eta, the real emission matrix, also time-inhomogenenous
    alpha = zeros(Nstate, Nstate, Nt);
    eta = zeros(Nstate, Nout, Nt);
    for t = 1: Nt
%         alpha(:,:,t) = exp(sum(K1.*repmat(reshape(S_rescale(:,t),[1,1,Nstim]),[Nstate,Nstate,1]),3) ...
%             + sum(H1.*repmat(reshape(Y_rescale(:,t:t+Ntau-1),[1,1,Ntau]),[Nstate,Nstate,1]),3));
        %rectified linear
        alpha(:,:,t) = sum(K1.*repmat(reshape(S_rescale(:,t),[1,1,Nstim]),[Nstate,Nstate,1]),3) ...
            + sum(H1.*repmat(reshape(Y_rescale(:,t:t+Ntau-1),[1,1,Ntau]),[Nstate,Nstate,1]),3);
        alpha(find(alpha<0)) = 0;
        alpha(:,:,t) = alpha(:,:,t)./sum(alpha(:,:,t),2);
%         eta(:,:,t) = exp(sum(K2.*repmat(reshape(S_rescale(:,t),[1,1,Nstim]),[Nstate,Nout,1]),3) ...
%             + sum(H2.*repmat(reshape(Y_rescale(:,t:t+Ntau-1),[1,1,Ntau]),[Nstate,Nout,1]),3));
        %rectified linear
        eta(:,:,t) = sum(K2.*repmat(reshape(S_rescale(:,t),[1,1,Nstim]),[Nstate,Nout,1]),3) ...
            + sum(H2.*repmat(reshape(Y_rescale(:,t:t+Ntau-1),[1,1,Ntau]),[Nstate,Nout,1]),3);
        eta(find(eta<0)) = 0;
        eta(:,:,t) = eta(:,:,t)./sum(eta(:,:,t),2);
    end
    
    % forward-backward
    %%% WRONG!!!
    [a, b, s] = forwardbackward(Y(Ntau + 1: end),alpha,eta); 
    Q2 = a.*b/sum(a(:,end));
    Q1 = zeros(Nstate, Nstate, Nt + 1);
    for t =1 : Nt 
        for n = 1: Nstate 
            for m = 1: Nstate
                Q1(n,m,t) = a(n,t)*alpha(n,m,t).*eta(m,Y(t + Ntau),t).*b(m,t+ 1); 
            end
        end
    end
    Q1 = Q1/sum(a(:,end));
    %% maximization
    % gradient ascent
    dK1 = zeros(Nstate,Nstate,Nstim);
    dH1 = zeros(Nstate,Nstate,Ntau);
    dK2 = zeros(Nstate,Nout,Nstim);
    dH2 = zeros(Nstate,Nout,Ntau);
    for t = 1: Nt
        for n = 1: Nstate
            for m = 1: Nstate
                if m ~= n
%                     dK1(n,m,:) = dK1(n,m,:) + reshape(Q1(n,m,t)*(1-alpha(n,m,t))*S_rescale(:,t),[1,1,Nstim]) ;
%                     dH1(n,m,:) = dH1(n,m,:) + reshape(Q1(n,m,t)*(1-alpha(n,m,t))*Y_rescale(:,t:t+Ntau-1),[1,1,Ntau]);
                    dK1(n,m,:) = dK1(n,m,:) + reshape(Q1(n,m,t)*(1-alpha(n,m,t))*S_rescale(:,t),[1,1,Nstim])* ...
                        max([sum(reshape(K1(n,m,:),[Nstim,1]).*S_rescale(:,t)) + sum(reshape(H1(n,m,:),[Ntau,1]).*Y_rescale(:,t:t+Ntau-1)),0]);
                    dH1(n,m,:) = dH1(n,m,:) + reshape(Q1(n,m,t)*(1-alpha(n,m,t))*Y_rescale(:,t:t+Ntau-1),[1,1,Ntau])*...
                        max([sum(reshape(K1(n,m,:),[Nstim,1]).*S_rescale(:,t)) + sum(reshape(H1(n,m,:),[Ntau,1]).*Y_rescale(:,t:t+Ntau-1)),0]);
                end
            end
            dK2(n,Y(t+Ntau),:) = dK2(n,Y(t+Ntau),:) + reshape(Q2(n,t+1)*(1-eta(n,Y(t+Ntau),t))*S_rescale(:,t),[1,1,Nstim])*...
                max([sum(reshape(K2(n,Y(t+Ntau),:),[Nstim,1]).*S_rescale(:,t)) + sum(reshape(H2(n,Y(t+Ntau),:),[Ntau,1]).*Y_rescale(:,t:t+Ntau-1)),0]);
            dH2(n,Y(t+Ntau),:) = dH2(n,Y(t+Ntau),:) + reshape(Q2(n,t+1)*(1-eta(n,Y(t+Ntau),t))*Y_rescale(:,t:t+Ntau-1),[1,1,Ntau])*...
                max([sum(reshape(K2(n,Y(t+Ntau),:),[Nstim,1]).*S_rescale(:,t)) + sum(reshape(H2(n,Y(t+Ntau),:),[Ntau,1]).*Y_rescale(:,t:t+Ntau-1)),0]);
        end        
    end
    K1 = K1+ dK1*rate;
    H1 = H1 + dH1*rate;
    K2 = K2 + dK2*rate;
    H2 = H2 + dH2*rate;
%     disp([sum(sum(isnan(b)))]);
    L(1,iter) = sum(log(s));    
    
    if mod(iter,1000)==0
    disp(['Iteration ',num2str(iter),', L = ',num2str(L(iter))]);
    
%     disp([max(max(max(dK1))),max(max(max(dH1))),max(max(max(dH2))),max(max(max(dH2)))]);
%     disp([min(min(min(K1))),min(min(min(H1))),min(min(min(H2))),min(min(min(H2)))]);
    end
    
    %try rescale the filters... hope it will work
    KH1_max = max([max(max(max(K1))), max(max(max(H1)))]);
    K1 = K1/KH1_max;
    H1 = H1/KH1_max;
    
    KH2_max = max([max(max(max(K2))), max(max(max(H2)))]);
    K2 = K2/KH2_max;
    H2 = H2/KH2_max;
end




























