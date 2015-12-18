
%% Train PCA

conv3_1=h5read('data_conv3_1.h5','/DS');
conv4_1=h5read('data_conv4_1.h5','/DS');
conv5_1=h5read('data_conv5_1.h5','/DS');
size(conv3_1)
size(conv4_1)
size(conv5_1)

conv3_1=reshape(conv3_1,[],size(conv3_1,4));
conv4_1=reshape(conv4_1,[],size(conv4_1,4));
conv5_1=reshape(conv5_1,[],size(conv5_1,4));
F=cat(1,conv3_1,conv4_1,conv5_1);
size(F)
clear conv3_1 conv4_1 conv5_1;
fprintf([sprintf('Memory Usage: %f GB',monitor_memory_whos/1024/1024/1024) '\n'])

fprintf('Starting PCA ...\n')
tic; [U,T,mu]=pcaecon(F,size(F,2)); toc
size(U)
size(T)
size(mu)
fprintf([sprintf('Memory Usage: %f GB',monitor_memory_whos/1024/1024/1024) '\n'])

delete('pca3_U.h5'); h5create('pca3_U.h5','/DS',size(U)); h5write('pca3_U.h5','/DS',U);
delete('pca3_T.h5'); h5create('pca3_T.h5','/DS',size(T)); h5write('pca3_T.h5','/DS',T);
delete('pca3_mu.h5'); h5create('pca3_mu.h5','/DS',size(mu)); h5write('pca3_mu.h5','/DS',mu);

%% Read data

tic;
U=h5read('pca_U.h5','/DS');
size(U)
T=h5read('pca_T.h5','/DS');
size(T)
mu=h5read('pca_mu.h5','/DS');
size(mu)
T(1:5,1)
toc

%% Read source, target and z-score features

source=h5read('out_source.h5','/DS')+1;
target=h5read('out_target.h5','/DS')+1;
fprintf(['source ' sprintf('%d ',source(1:5)) '$\n'])
fprintf(['target ' sprintf('%d ',target(1:5)) '$\n'])

feat_source=T(:,source);
feat_target=T(:,target);
feat_img=T(:,1);

[min(feat_img) max(feat_img)]
[min(feat_source(:)) max(feat_source(:))]
[min(feat_target(:)) max(feat_target(:))]

feat_all = [feat_source feat_target]; 
[min(feat_all(:)) max(feat_all(:))]
s = std(feat_all(:)); 
mee = mean(feat_all(:));
s
mee

feat_img = (feat_img - mee) / s;    
feat_source = (feat_source - mee) / s;
feat_target = (feat_target - mee) / s; 
[min(feat_img) max(feat_img)]
[min(feat_source(:)) max(feat_source(:))]
[min(feat_target(:)) max(feat_target(:))]

recon_feat = U*((feat_img * s) + mee) + mu;
[min(recon_feat(:)) max(recon_feat(:))]

%% Solve for r

sqrsig = 1e4; 
lambdas = [5e-6];

d = length(feat_img); 
r = zeros(d,1); %/1e4;
for i = 1:length(lambdas),
    tic; [ropt,fx,iters] = minimize(r,'witness_obj2',50,feat_img,feat_source,feat_target,sqrsig,lambdas(i)); toc
    new = feat_img + ropt;
    new = U*((new * s) + mee) + mu;
    transformed_feat{i} = new;
    fprintf([sprintf('||r|| = %f',(ropt'*ropt)^0.5) '\n'])
    [min(new) max(new)]
    new(1:10)
end

%% Save results for reconstruction

conv3_1=reshape(new(1:32*32*256),[32 32 256]);
conv4_1=reshape(new(32*32*256+1:32*32*256+16*16*512),[16 16 512]);
conv5_1=reshape(new(32*32*256+16*16*512+1:32*32*256+16*16*512+8*8*512),[8 8 512]);
if ndims(conv3_1)==3
  % workaround MATLAB bug where trailing singleton dims are dropped
  conv3_1=repmat(conv3_1,[1 1 1 2]);
  conv4_1=repmat(conv4_1,[1 1 1 2]);
  conv5_1=repmat(conv5_1,[1 1 1 2]);
end
size(conv3_1)
size(conv4_1)
size(conv5_1)

delete('test_r.h5'); h5create('test_r.h5','/DS',size(ropt,1)); h5write('test_r.h5','/DS',ropt);
delete('test_conv3_1.h5'); h5create('test_conv3_1.h5','/DS',size(conv3_1)); h5write('test_conv3_1.h5','/DS',conv3_1);
delete('test_conv4_1.h5'); h5create('test_conv4_1.h5','/DS',size(conv4_1)); h5write('test_conv4_1.h5','/DS',conv4_1);
delete('test_conv5_1.h5'); h5create('test_conv5_1.h5','/DS',size(conv5_1)); h5write('test_conv5_1.h5','/DS',conv5_1);

%% Reconstruct

% Run the reconstruction outside MATLAB (because LD_LIBRARY_PATH causes a name conflict)
%system('./gen_deepart.py reconstruct --test_indices=[0] --data_indices=[0] --prefix=test --desc=testmatlab')

