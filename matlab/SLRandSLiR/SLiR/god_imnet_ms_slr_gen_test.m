function [res P stats]= god_imnet_ms_slr_gen_test(res,stats,D_tr,D_te,pp,P,P_te,suffix)
%
%

%% Set parameters:
P.script_name = mfilename;
P.date_time   = datestr(now,'yyyy-mm-dd HH:MM:SS');

%% ----------------------------------------------------------------------------

%% Reordering
% original order is alphabetical order of images
% reordering to presentation order
visfeat=res.data.trtrue;
visfeat_test=res.data.tetrue;

%% parameter settings
[nsample,nfeat]=size(visfeat);
[nsample_test,nfeat_test]=size(visfeat_test);
nvox=size(D_tr.data,2)+1;

% crossvalidation parameters
% ncv=14;
% ncv=2;
%nsample_in1cv=nsample/ncv;
%indm=makeM(1:nsample,[nsample_in1cv,ncv],1);
% error
err_tr=zeros(nfeat,1);
cor_tr=zeros(nfeat,1);
err_te=zeros(nfeat,1);
cor_te=zeros(nfeat,1);

% setting for sparse linear regression analysis
parm.Ntrain = 2000; % # of total training iteration
parm.Nskip  = 2000;     % skip steps for display info
parm.data_norm =1;

% training data
trpred=cell(nfeat,1);
trtrue=cell(nfeat,1);
% CV test data
% tepred=zeros(nfeat,nsample);
tepred=cell(nfeat,1);
tetrue=cell(nfeat,1);
norm.xmean=cell(nfeat,1);
norm.xnorm=cell(nfeat,1);
norm.ymean=cell(nfeat,1);
norm.ynorm=cell(nfeat,1);
%% prediction
weights=zeros(nvox,nfeat);
wfreq=cell(nfeat,1);
tic
for itr=1:nfeat
    %     for iitr=1:nsample
    wcv=zeros(nvox,1);
    %for iitr=1:ncv
    fprintf('Prediction for %3d/%4dth visualword------------------------------\n',itr,nfeat)
    fprintf('[%2dd %2dh %2dm %2ds]:\n',floor(toc/86400),mod(floor(toc/3600),24),mod(floor(toc/60),60),floor(mod(toc,60)))
    !hostname
    fprintf('%s\n',suffix)
    train_x=D_tr.data;
    test_x=D_te.data;
    train_y=visfeat(:,itr);
    test_y=visfeat_test(:,itr);

    %% normalization-------
    [train_x,nparm]=normalize_data(train_x',parm.data_norm);
    parm.xmean=nparm.xmean;
    parm.xnorm=nparm.xnorm;
    [train_y,nparm]=normalize_data(train_y',parm.data_norm);
    parm.ymean=nparm.xmean;
    parm.ynorm=nparm.xnorm;
    test_x=normalize_data(test_x',parm.data_norm,parm);
    norm.xmean{itr}=parm.xmean;
    norm.xnorm{itr}=parm.xnorm;
    norm.ymean{itr}=parm.ymean;
    norm.ynorm{itr}=parm.ynorm;
    %% --------------------

    % model training-----
    Model=stats.Model{itr};
    wcv(Model.ix_act,1)=Model.W';
    wfreq{itr,1}=Model.ix_act;
    % --- Prediction for training data
    trpred{itr}(:,1) = predict_output(addBias(train_x')', Model, parm);
    train_y=visfeat(:,itr);
    trtrue{itr}(:,1) = train_y;
    err_tr(itr,1)=sum((trpred{itr}(:,1)-train_y).^2)/sum(train_y.^2);% nRMSE?
    cor_tr(itr,1)=fcorr(trpred{itr}(:,1),train_y);
    % --- Prediction for test data
    tepred{itr}(:,1) = predict_output(addBias(test_x')', Model, parm);
    tetrue{itr}(:,1) = test_y;

    %end
    weights(:,itr)=mean(wcv,2);
    err_te(itr)=sum((tepred{itr}(:)-visfeat_test(:,itr)).^2)/sum(visfeat_test(:,itr).^2);% nRMSE?
    cor_te(itr)=fcorr(tepred{itr}(:),visfeat_test(:,itr));
    %     [(1:itr)',mean(err_tr(1:itr,:),2),mean(cor_tr(1:itr,:),2),err_te(1:itr),cor_te(1:itr)]
end
w.vals=weights;
w.freq=wfreq;
w.volInds=D_tr.volInds;

%% results
dat_trp_tmp=zeros(length(trpred{1}(:)),nfeat);
dat_trt_tmp=zeros(length(trpred{1}(:)),nfeat);
for itr=1:nfeat
    dat_trp_tmp(:,itr)=trpred{itr}(:);
    dat_trt_tmp(:,itr)=trtrue{itr}(:);
end
dcor_tr=diag(fcorr(dat_trt_tmp',dat_trp_tmp'));

dat_tep_tmp=zeros(size(visfeat_test));
dat_tet_tmp=zeros(size(visfeat_test));
for itr=1:nfeat
    dat_tep_tmp(:,itr)=tepred{itr}(:);
    dat_tet_tmp(:,itr)=tetrue{itr}(:);
end
dcor_te=diag(fcorr(dat_tet_tmp',dat_tep_tmp'));

%% summary
clear res
res.train.err=err_tr;
res.train.cor_s=cor_tr;
res.train.cor_d=dcor_tr;
res.test.err=err_te;
res.test.cor_s=cor_te;
res.test.cor_d=dcor_te;

res.data.trtrue=visfeat;
% res.data.trtrue=trtrue;
res.data.trpred=trpred;
res.data.tetrue=visfeat_test;
% res.data.tetrue=tetrue;
if iscell(tepred)
    tmp=zeros(nsample_test,nfeat);
    for itr=1:nfeat
        tmp(:,itr)=tepred{itr}(:);
    end
    res.data.tepred=tmp;
else
    res.data.tepred=tepred;
end
res.data.imgOrder=D_tr.labels;
res.data.imgOrder_test=D_te.labels;

clear stats
stats.w=w;
stats.norm=norm;




%% end
end
