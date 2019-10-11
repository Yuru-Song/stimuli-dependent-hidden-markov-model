function alpha = compute_trans(S, F)
% INPUT: 
%   S: [num_feat, num_time]
%   F: [num_state, num_state, num_feat + 1]
% OUTPUT:
%   alpha: [num_state, num_state, num_time]

% extract dimensions
[num_feat, num_time] = size(S);
S = [S; ones(1, num_time)];
num_state = size(F, 1);
alpha = zeros(num_state, num_state, num_time);
for t = 1 : num_time
    alpha(:, :, t) = exp(sum(F.*repmat(reshape(S(:,t),[1, 1, num_feat + 1]),[num_state, num_state]),3));
    alpha(:, :, t) = alpha(:, :, t) ./ sum(alpha(:, :, t), 2);
    if any(isnan(alpha(:,:,t)))
        error(['NaN in alpha(:,:,t), t = ', num2str(t)]);
    end
end