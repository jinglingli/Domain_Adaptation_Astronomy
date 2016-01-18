
function [Mt,idf] = tfidf(M)

% Input:
% M: dxn 
% Mt: dxn

[d, n] = size(M);

temp = zeros(1, n);
all = sum(M,1);
temp(all > 0) = 1./all(all > 0);

tf = M.*repmat(temp, d, 1);

idf = log(n./(sum(M>0, 2)));
idf(sum(M>0, 2)==0)=0;

Mt = tf.*repmat(idf, 1, n);

