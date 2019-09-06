% SH_hmmgenerate generate a sequence for a StimHistHMM
% ---------------------------------------------------
% 	INPUTS:
% 		L: integer, length of sequence to generate
% 		K1: [numStates, numStates, numStim] array, linear filter of stimulus-dependent state transition 
% 		H1: [numStates, numStates, numTau] array, linear filter of history-dependent state transition
% 		K2: [numStates, numEmiss, numStim] array, linear filter of stimulus-dependent emission
% 		H2: [numStates, numEmiss, numTau] array, linear filter of history-dependent emission
% 		stim: [numStim, L] array, stimuli over time
% 	OUTPUTS:
% 		seq: [1, L] array
% 		states: [1, L] array
% ---------------------------------------------------
% reference: Hidden Markov models for the stimulus-response relationships of multi-state neural systems, Escola et al, 
% Neural Computation, 2010
%	See also  forwardbackward, trainHH, emisshessian.
% ---------------------------------------------------
% @Yuru Song, Aug-31-2019, UCSD
function [Y, states, alpha, eta] = generate(Nt, K1, H1, K2, H2, S)
Nstate = size(K1,1);
Nstim = size(K1,3);
Ntau = size(H1,3);
Nout = size(K2, 2);
states = zeros(1,Nt);
Y = zeros(1,Nt + Ntau);
% create two random sequences, one for state changes, one for emission
statechange = rand(1,Nt);
randvals = rand(1,Nt);
% Assume that we start in state 1.
currentstate = 1;
alpha = zeros(Nstate,Nstate,Nt);
eta = zeros(Nstate,Nout,Nt);
for t = 1: Nt
    % stimuli and history dependent
    % time-heterogeneous transition matrix
    alpha(:,:,t) = exp(sum(K1.*repmat(reshape(S(:,t),[1,1,Nstim]),[Nstate,Nstate,1]),3) ...
            + sum(H1.*repmat(reshape(Y(:,t:t+Ntau-1),[1,1,Ntau]),[Nstate,Nstate,1]),3));
    alpha(:,:,t) = alpha(:,:,t)./sum(alpha(:,:,t),2);
    alphac = cumsum(alpha(:,:,t),2);
    % time-heterogeneous emission matrix
    eta(:,:,t) = exp(sum(K2.*repmat(reshape(S(:,t),[1,1,Nstim]),[Nstate,Nout,1]),3) ...
            + sum(H2.*repmat(reshape(Y(:,t:t+Ntau-1),[1,1,Ntau]),[Nstate,Nout,1]),3));
    eta(:,:,t) = eta(:,:,t)./sum(eta(:,:,t),2);
    etac = cumsum(eta(:,:,t),2);
	% calculate state transition
    stateVal = statechange(t);
    state = 1;
    for innerState = Nstate-1:-1:1
        if stateVal > alphac(currentstate,innerState)
            state = innerState + 1;
            break;
        end
    end
    % calculate emission
    val = randvals(t);
    emit = 1;
    for inner = Nout-1:-1:1
        if val  > etac(state,inner)
            emit = inner + 1;
            break
        end
    end
    % add values and states to output
    Y(t + Ntau) = emit;
    states(t) = state;
    currentstate = state;
end
Y = Y(Ntau + 1:end);






