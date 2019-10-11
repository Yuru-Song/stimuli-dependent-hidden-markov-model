function eta = compute_emiss(S, G)
% INPUT: 
%   S: [num_feat, num_time]
%   G: [num_state, num_out, num_feat + 1]
% OUTPUT:
%   eta: [num_state, num_out, num_time]

% extract dimensions
[num_feat, num_time] = size(S);
num_state = size(G, 1);
num_out = size(G, 2);
S = [S; ones(1, num_time)];
eta = zeros(num_state, num_out, num_time);
for t = 1: num_time
    eta(:, :, t) = exp(sum(G.*repmat(reshape(S(:, t), [1, 1, num_feat + 1]), [num_state, num_out]), 3));
    eta(:, :, t) = eta(:, :, t)./sum(eta(:, :, t), 2);
    if any(isnan(eta(:,:,t)))
        error(['NaN in eta(:,:,t), t = ', num2str(t)]);
    end    
end