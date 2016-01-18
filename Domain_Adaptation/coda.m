
function [W, loss, acc, pred]  = coda(dataX, labels, idxLabs, idxUnls, idxTest, pos, neg, lambda, maxIter, epsilon, idxSS, idxTT, gamma, decreRatio)

% INPUT:
% dataX    dxn   all the inputs
% labels   1xn	 labels for all the inputs
% idxLabs  1xnTr indices of labeled inputs
% idxUnls  1xnU  indices of unlabeled inputs
% idxTest  1xnTe indices of test inputs
% pos, neg number of positive/negative examples to add to the labeled set in each round
% lambda   l2 regularizer
% maxIter  maximum number of iterations of co-training
% epsilon  epsilon expanding condition
% idxSS    indices of the source data
% idxTT    indices of the target unlabeled data
% gamma    l1 regularizer 
% decreRatio discount the l1 regularizer in every iteration


nClass = length(unique(labels(idxLabs)));

[d,n] = size(dataX);

penalty = zeros(1,n);
penalty(idxLabs) = 1;

iter = 1;
acc = zeros(maxIter, 3);
pred = zeros(1, size(idxTest,2));

xTe = dataX(:,idxTest);
yTe = labels(idxTest);

dataY = zeros(1,n);
dataY(idxLabs) = labels(idxLabs);

idxA = idxLabs;
idxB = idxLabs;
nTr = length(idxLabs);
backup = idxUnls;

count = 0;

W = [];

splitInitial = randn(2*d, 1);
while size(idxUnls,2) >= 0 && iter <= maxIter
	assert( sum(dataY(idxLabs) ~= labels(idxLabs)) == 0);
	nUnls = size(idxUnls,2);
	count = count +1;

	bA = penalty(idxA)/length(idxA)/2;
	bB = penalty(idxB)/length(idxB)/2;
	converged = false;
	varA = [];
	varB = [];
	xSS = dataX(:, idxSS);
	xTT = dataX(:, idxTT);
	stdXSS = std(xSS, 1, 2);
	stdXTT = std(xTT, 1, 2);
	while true
        	[varA, varB, w, loss_split, converged] = feature_split(splitInitial, dataX(:,idxA), dataY(idxA),  bA, dataX(:, idxB), dataY(idxB), bB,  100, 0, dataX(:, idxUnls), epsilon, xSS, dataY(idxSS), xTT, stdXSS, stdXTT, gamma);
		gamma = gamma/decreRatio;
		if length(varA) > 0 && length(varB) > 0
			break;
		end
	end

	varA = [varA; d];
	varB = [varB; d];
	initial = zeros(length(varA), 1);
	[wwA,loss]= minimize(initial, @logistic, 100, dataX(varA,idxA), dataY(idxA), bA, lambda);
	wA = zeros(d, 1);
	wA(varA) = wwA;
	initial = zeros(length(varB), 1);
	[wwB,loss]= minimize(initial, @logistic, 100, dataX(varB,idxB), dataY(idxB), bB, lambda);
	wB = zeros(d, 1);
	wB(varB) = wwB;

	w = [wA; wB];
	W = [W w];

        [pred, acc(iter,:)] = predict(xTe, yTe, wA, wB);
	fprintf('Iter = %d, Classification Error A: %f, B: %f, combined: %f\n', iter, acc(iter, 1), acc(iter, 2), acc(iter, 3));
        confA = wA'*dataX(:,idxUnls);
        confB = wB'*dataX(:,idxUnls);

        beliefA = 1./(1+exp(-abs(confA)));
        beliefB = 1./(1+exp(-abs(confB)));
    	idx = find( abs(confA) > abs(confB) & beliefA > 0.8 & sign(confA) == sign(confB) );
    	plusB = [idx; abs(confA(idx)); sign(confA(idx)); abs(confA(idx)) - abs(confB(idx)) ];
    	idx = find( abs(confB) > abs(confA)  & beliefB > 0.8 & sign(confA) == sign(confB) );
    	plusA = [idx; abs(confB(idx)); sign(confB(idx)); abs(confB(idx)) - abs(confA(idx))];

        if (size(plusA,2) == 0 && size(plusB, 2) == 0)
		fprintf('no more confidence predictions, stop at iteration %d\n', iter);
                break;
        end

        setA1 = [];
        setB1 = [];
        setA2 = [];
        setB2 = [];
        if size(plusA,2) > 0
                plusA1 = plusA(1:4, plusA(3,:) > 0);
                plusA2 = plusA(1:4, plusA(3,:) < 0);

		%'AddToAPlus'
		if length(plusA1(4,:)) < 10000
                [sorted, ind] = sort(plusA1(4,:));
                if (size(ind,2) > pos)
                        ind = ind(end-pos+1:end);
                end
		else
			ind=zeros(1,pos);
			for tt=1:pos
				[mm,ind(tt)]=max(plusA1(4,:));
				plusA1(4,ind(tt))=0;	
			end
		end
                idx = idxUnls(plusA1(1,ind));
                idxA = [idxA idx];
                penalty(idx) = 1./(1+exp(-plusA1(2,ind)));
                dataY(idx) = 1;
                setA1 = plusA1(1,ind);
	
		%'AddToAMinus'
		if length(plusA2(4,:)) < 10000
                [sorted, ind] = sort(plusA2(4,:));
                if (size(ind,2) > neg)
                        ind = ind(end-neg+1:end);
                end
		else
                        ind=zeros(1,neg);
                        for tt=1:neg
                                [mm,ind(tt)]=max(plusA2(4,:));
                                plusA2(4,ind(tt))=0;   
                        end
		end
                idx = idxUnls(plusA2(1,ind));
                idxA = [idxA idx];
                penalty(idx) = 1./(1+exp(-plusA2(2,ind)));
                dataY(idx) = -1;
                setA2 = plusA2(1,ind);
        end

        if size(plusB,2) > 0
                plusB1 = plusB(1:4, plusB(3,:) > 0);
                plusB2 = plusB(1:4, plusB(3,:) < 0);

		%'AddToBPlus'
		if length(plusB1(4,:)) < 10000
                [sorted, ind] = sort(plusB1(4,:));
                if (size(ind,2) > pos)
                        ind = ind(end-pos+1:end);
                end
		else
                        ind=zeros(1,pos);
                        for tt=1:pos
                                [mm,ind(tt)]=max(plusB1(4,:));
                                plusB1(4,ind(tt))=0;     
                        end     
		end
                idx = idxUnls(plusB1(1,ind));
                idxB = [idxB idx];
                penalty(idx) = 1./(1+exp(-plusB1(2,ind)));
                dataY(idx) = 1;
                setB1 = plusB1(1,ind);

		%'AddToBMinus'
		if length(plusB2(4,:)) < 10000
                [sorted, ind] = sort(plusB2(4,:));
                if (size(ind,2) > neg)
                        ind = ind(end-neg+1:end);
                end
		else
                        ind=zeros(1,neg);
                        for tt=1:neg
                                [mm,ind(tt)]=max(plusB2(4,:));
                                plusB2(4,ind(tt))=0;
                        end
		end
                idx = idxUnls(plusB2(1,ind));
                idxB = [idxB idx];
                penalty(idx) = 1./(1+exp(-plusB2(2,ind)));
                dataY(idx) = -1;
                setB2 = plusB2(1,ind);
        end

	idxSS = [idxSS idxUnls(unique([setA1 setA2 setB1 setB2]))];
	idxUnls(unique([setA1 setA2 setB1 setB2])) = 0;
        idxUnls = idxUnls(idxUnls > 0);
	idxTT = idxUnls;

        iter = iter+1;
end
loss = loss_split;

