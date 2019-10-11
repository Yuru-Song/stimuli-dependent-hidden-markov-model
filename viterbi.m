function [currentState, logP] = viterbi(seq,tr,e)
%VITERBI calculates the most probable state path for a sequence, in StimHMM model.
% INPUT: 
%     seq: [1, num_time]
%     tr: [num_state, num_state, num_time]
%     e: [num_state, num_out, num_time]
% OUTPUT:
%     currentState: the most likely paths 
% @ Oct-01-2019, Yuru Song


if nargin > 0
    if isstring(seq)
        seq = cellstr(seq);
    end
end

numStates = size(tr,1);
checkTr = size(tr,2);
if checkTr ~= numStates
    error('BadTransitions');
end
% number of rows of e must be same as number of states

checkE = size(e,1);
if checkE ~= numStates
    error('InputSizeMismatch');
end

numEmissions = size(e,2);
customStatenames = false;
% work in log space to avoid numerical issues
L = length(seq);
if any(seq(:)<1) || any(seq(:)~=round(seq(:))) || any(seq(:)>numEmissions)
     error('BadSequence');
end
currentState = zeros(1,L);
if L == 0
    return
end
logTR = log(tr);
logE = log(e);
% allocate space
pTR = zeros(numStates,L);
% assumption is that model is in state 1 at step 0
v = -Inf(numStates,1);
v(1,1) = 0;
vOld = v;

% loop through the model
for count = 1:L
    for state = 1:numStates
        % for each state we calculate
        % v(state) = e(state,seq(count))* max_k(vOld(:)*tr(k,state));
        bestVal = -inf;
        bestPTR = 0;
        % use a loop to avoid lots of calls to max
        for inner = 1:numStates 
            val = vOld(inner) + logTR(inner,state,count);
            if val > bestVal
                bestVal = val;
                bestPTR = inner;
            end
        end
        % save the best transition information for later backtracking
        pTR(state,count) = bestPTR;
        % update v
        v(state) = logE(state,seq(count), count) + bestVal;
    end
    vOld = v;
end

% decide which of the final states is post probable
[logP, finalState] = max(v);

% Now back trace through the model
currentState(L) = finalState;
for count = L-1:-1:1
    currentState(count) = pTR(currentState(count+1),count+1);
    if currentState(count) == 0
        error('ZeroTransitionProbability');
    end
end
if customStatenames
    currentState = reshape(statenames(currentState),1,L);
end


