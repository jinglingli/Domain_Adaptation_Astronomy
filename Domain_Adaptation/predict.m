function [newpred, accuracy] = predict(xTe, yTe, w1, w2)

% INPUT:
% xTe dxn matrix
% yTe 1xn matrix
% w1 dx1 weight vector for the first classifier
% w2 dx1 weight vector for the second classifier 

conf = [w1'*xTe; w2'*xTe; (w1+w2)'*xTe];
pred = sign(conf);
newpred = pred(3,:);
accuracy = sum(pred ~= repmat(yTe, 3 , 1), 2)/length(yTe);
accuracy = accuracy';
