function [loss,gradient]= computeLossGrad(w, xTrA, yTrA, beliefA, xTrB, yTrB, beliefB, lambda, xU, epsilon, xSS, ySS, xTT, stdXSS, stdXTT, gamma, alpha, c, alpha2, c2)
% INPUT:
% w 2*d weight vector
% xTrA    dx(nA) the training set for the first classifier
% yTrA    1x(nA) the labels
% beliefA 1x(nA) the weights
% xTrB    dx(nB) the training set for the second classifier
% yTrB    1x(nB) the labels
% beliefB 1x(nB) the weights
% lambda  l2 regularizer for logistic regression
% xU      dx(nU) the unlabeled set
% epsilon hyperparameter for balcan's epsilon expanding theory
% alpha, c, alpha2, c2    lagrangian multipliers for the augmented lagrangian methods
%



d = size(xTrA, 1);

w1 = w(1:d-1);
w2 = w(d+1:end-1);


overlap = sum(w1.*w1.*w2.*w2);
loss = alpha*overlap + c/2*overlap*overlap;
gradient = 2*[w1.*w2.*w2; 0; w2.*w1.*w1; 0]*(alpha + c*overlap);

[loss1, grad1] = logistic(w(1:d), xTrA, yTrA, beliefA, lambda);
[loss2, grad2] = logistic(w(d+1:end), xTrB, yTrB, beliefB, lambda);


if (loss1 > loss2)
	loss = loss + loss1 + log(1+exp(loss2-loss1));
else
	loss = loss + loss2 + log(1+exp(loss1-loss2));
end
gradient = gradient + [1/(1+exp(loss2-loss1))*grad1; 1/(exp(loss1-loss2)+1)*grad2];



[l, grad] = balcanExpansion(w, xU, epsilon);
loss = loss + alpha2*l + c2/2*l*l;
gradient = gradient + grad*(alpha2+c2*l);



%[l, grad] = incompatibility(w, xSS, xTT, ySS, stdXSS, stdXTT);
[l, grad] = covv2(w, xSS, xTT, ySS, stdXSS, stdXTT);
loss = loss + gamma*l;
gradient = gradient + gamma*grad;
