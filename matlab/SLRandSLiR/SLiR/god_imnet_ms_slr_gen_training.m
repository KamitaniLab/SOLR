function [res P stats]= god_imnet_ms_slr_gen_training(D_tr,D_te,visfeat,visfeat_test,pp,P,P_te,suffix)
%
%

%% Set parameters:
P.script_name = mfilename;
P.date_time   = datestr(now,'yyyy-mm-dd HH:MM:SS');

%% ----------------------------------------------------------------------------

%% Reordering
% original order is alphabetical order of images
% reordering to presentation order
imageVarInd=ismember(D_tr.labels_type,'ImageVar');
visfeat=double(visfeat(D_tr.labels(:,imageVarInd),:));
visfeat(visfeat(:)==0)=rand(sum(visfeat(:)==0),1)*1e-100;

visfeat_test=double(visfeat_test(1:size(D_te.data,1),:));
visfeat_test(visfeat_test(:)==0)=rand(sum(visfeat_test(:)==0),1)*1e-100;

%% parameter settings
[nsample,nfeat]=size(visfeat);
[nsample_test,nfeat_test]=size(visfeat_test);
nvox=size(D_tr.data,2)+1;

% crossvalidation parameters
% ncv=14;
% ncv=2;
%nsample_in1cv=nsample/ncv;
%indm=makeM(1:nsample,[nsample_in1cv,ncv],1);

% setting for sparse linear regression analysis
parm.Ntrain = 2000; % # of total training iteration
parm.Nskip  = 2000;     % skip steps for display info
parm.data_norm =1;

norm.xmean=cell(nfeat,1);
norm.xnorm=cell(nfeat,1);
norm.ymean=cell(nfeat,1);
norm.ynorm=cell(nfeat,1);
%% prediction
tic
Model=cell(nfeat,1);
for itr=1:nfeat
    %     for iitr=1:nsample
    %wcv=zeros(nvox,1);
    %for iitr=1:ncv
    fprintf('Prediction for %3d/%4dth visualword------------------------------\n',itr,nfeat)
    fprintf('[%2dd %2dh %2dm %2ds]:\n',floor(toc/86400),mod(floor(toc/3600),24),mod(floor(toc/60),60),floor(mod(toc,60)))
    !hostname
    fprintf('%s\n',suffix)
    train_x=D_tr.data;
    %test_x=D_te.data;
    train_y=visfeat(:,itr);
    %test_y=visfeat_test(:,itr);

    %% normalization-------
    [train_x,nparm]=normalize_data(train_x',parm.data_norm);
    parm.xmean=nparm.xmean;
    parm.xnorm=nparm.xnorm;
    [train_y,nparm]=normalize_data(train_y',parm.data_norm);
    parm.ymean=nparm.xmean;
    parm.ynorm=nparm.xnorm;
    %test_x=normalize_data(test_x',parm.data_norm,parm);
    %norm.xmean{itr}=parm.xmean;
    %norm.xnorm{itr}=parm.xnorm;
    %norm.ymean{itr}=parm.ymean;
    %norm.ynorm{itr}=parm.ynorm;
    %% --------------------

    % model training-----
    Model{itr} = linear_map_sparse_cov(addBias(train_x')',train_y, Model{itr}, parm);
end

%% summary
res.data.trtrue=visfeat;
res.data.tetrue=visfeat_test;

stats.Model=Model;
%stats.norm=norm;




%% end
end
