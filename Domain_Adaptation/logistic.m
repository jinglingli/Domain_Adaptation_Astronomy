
function [loss,gradient]=logistic(w,xTr,yTr,belief,lambda)
%
% INPUT:
% w weight vector
% xTr dxn matrix
% yTr 1xn matrix 
% belief 1xn matrix (weights for each input)
% lambda l2 regularization factor
%

ewxy=exp(-w'*xTr.*yTr);
loss=sum(log(1+ewxy).*belief) + lambda/2*w(1:end-1)'*w(1:end-1);
gradient=-xTr*(ewxy.*yTr.*belief./(1+ewxy))' + lambda*[w(1:end-1);0];

