coda_addpos  = 10; 
coda_addneg  = 10;
coda_lambda  = 1e-16; % logistic l2 regularizer
coda_gamma   = 0.0001; % feature incompatibility
coda_epsilon = 1.0;
coda_decrat  = 1.00001;
coda_niter   = 250;

coda_X = [Xstr; Xttrl; Xttru; Xtte]';
coda_X = [coda_X; ones(1,size(coda_X,2))]; % add bias term

coda_Y = [ystr; yttrl; yttru; ytte]';

% coda assumes labels \in {-1,1} change class label 0 to -1 
coda_Y(coda_Y==0) = -1;

% labeled/unlabeled indices
coda_idxL = 1:(nstr+nttrl);  
coda_idxU = (nstr+nttrl)+(1:nttru); 

% src/tgt indices
coda_idxSS = 1:nstr;  
coda_idxTT = nstr+(1:(nttrl+nttru+ntte)); 

% target test indices
coda_idxTest = (nstr+nttrl+nttru)+(1:ntte);          

[coda_W, coda_loss, coda_acc, coda_pred, coda_fpr, coda_fnr, coda_iter] = ...
    coda(coda_X, coda_Y, coda_idxL, coda_idxU, coda_idxTest, ...
         coda_addpos, coda_addneg, coda_lambda, coda_niter, ...
         coda_epsilon, coda_idxSS, coda_idxTT, coda_gamma, coda_decrat);    
% coda acc is err per iteration for a, b and combined cases (niter x 3 array)
%accj = 1-coda_acc(max(find(coda_acc(:,3))),3);

% coda preds in {-1,1}, replace -1 with 0
coda_pred(coda_pred==-1) = 0;
accuracy = mean(coda_pred'==ytte);