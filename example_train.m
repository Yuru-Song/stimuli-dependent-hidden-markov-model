clear all;
rng(1);

num_time =200;
num_feat = 10;
num_seq = 1;
num_state = 2;
num_out = 3;
num_train = 1;

% options.num_state = 2;
% options.verbose = 1;
% options.tol = 1e-20;
% options.max_iter =100;
% options.disp_step = 10;
% % options.F =  reshape([[	0],[1];[log(7/3)],[0]], [num_state, num_state, num_feat]);
% % options.G = reshape([[0],[1],[1]; [0],[1],[1]], [num_state, num_out, num_feat]);
% options.debug = 0;
% S = zeros(num_seq,num_feat, num_time);
% S(:, end, :) = 1;
tr = [0,1;0.7,.3];%state 1: want, state 2: don't want
e = [.8,.1,.1;.1,.2,.7];%output 1: court, 2: move away, 3: still

% F = zeros( num_state, num_state, num_feat);
% G = zeros(num_state, num_out, num_feat);
% ecll = zeros(1,options.max_iter);
Y = zeros(num_seq, 1, num_time);
[Y(1, 1, :), emiss_filter, trans_filter, features] = generate(num_feat, num_state, num_out, num_time);
S(1, :,:) = features;
options.G = emiss_filter;
options.F = trans_filter;
for seq = 2: num_seq
    [Y(seq, 1, :), emiss_filter, trans_filter, features] = generate(num_feat, num_state, num_out, num_time, options);
    S(seq, :,:) = features;
% [Y(seq, 1, :), ~] = hmmgenerate(num_time, tr, e);
end
% save_path =  '/Users/yurusong/Desktop/1_Work/1_Current_Projects/2_Fly_Behavior_UCSD/1_Workspace/StimHMM/example_train';
save_path =  '/home/yuru/example_train_StimHMM';
for train = 1: num_train
[F, G, ecll, U, V] = StimHMMtrain(S, Y);
% save([save_path, '/random_init_',num2str(train),'.mat'],'F','G','ecll','U','V','Y','S');
end


% figure(1);
% subplot(121)
% imagesc(tr)
% colorbar
% xticks([])
% yticks([])
% title('true trans matrix');
% subplot(122)
% imagesc(alpha(:,:,end))
% colorbar
% xticks([])
% yticks([])
% title('estimated trans matrix')
% set(gcf,'Units','Normalized','OuterPosition',[0.,0.3,.35,.3])
% % saveas(1,['example_train/trans',num2str(seq)],'epsc');
% 
% figure(2);
% subplot(121)
% imagesc(e)
% colorbar
% title('true emission matrix');
% xticks([])
% yticks([])
% subplot(122)
% imagesc(eta(:,:,end))
% colorbar
% xticks([])
% yticks([])
% title('estimated emission matrix');
% set(gcf,'Units','Normalized','OuterPosition',[0.,.7,.35,.3])
% save([ '/Users/yurusong/Desktop/1_Work/1_Current_Projects/2_Fly_Behavior_UCSD/1_Workspace/StimHMM/example_train','/filters_2.mat'],'F','G','ecll','loglik')