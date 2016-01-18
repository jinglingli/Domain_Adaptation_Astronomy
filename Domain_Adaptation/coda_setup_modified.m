function pred = coda_setup_modified(args)

dataX = args.arg1;
labels = args.arg2;
idxLabs = args.arg3;
idxUnls = args.arg4;
idxTest = args.arg5;
idxSS = args.arg6;
idxTT = args.arg7;

dataX = [dataX; ones(1,size(dataX,2))]; % add bias term
pos = 10;
neg = 10;
gamma = 0.001;
lambda = 1e-16; % logistic l2 regularizer 1e-16
maxIter = 100;
epsilon = 1.0;
decreRatio=1.0005;

[W, loss, acc, pred] = coda(dataX, labels, idxLabs, idxUnls, idxTest, pos, neg, lambda, maxIter, epsilon, ...
idxSS, idxTT, gamma, decreRatio);

count = 0;
for i=size(acc,1):-1:1
    if acc(i)~=0
        count = i;
        break
    end
end
acc = acc(count);
pred = [pred acc]
fclose('all');