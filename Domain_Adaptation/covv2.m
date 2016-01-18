function [loss,gradient]= covv2(w, X1, X2, Y1, stdX1, stdX2)

% INPUT:
% dataU dxn matrix (each column is an input vector)
% w 2*d weight vector (default w=0)
[d, l1] = size(X1);
[d, l2] = size(X2);
Y1 = Y1';
s1 = (X1*Y1/l1 - mean(X1, 2)*mean(Y1) )./stdX1/std(Y1, 1);
barX2 = mean(X2, 2);
norm = max(abs(s1));

s1 = s1./stdX2;
s1(isnan(s1)) = 0;
s1(isinf(s1)) = 0;
w1 = w(1:d);
w2 = w(d+1:end);
h = (w1+w2)'*X2;
ewx = exp(-abs(h));
pplus = 1./(1+ewx);
Y2 = (sign(h).*(2*pplus-1))';
barY2 = mean(Y2);
stdY2 = std(Y2, 1);
s2 = (X2*Y2/l2 - barX2*barY2 )/stdY2;

s2(isnan(s2)) = 0;
u = w1(1:end-1);
v = w2(1:end-1);
n1 = sqrt(u.*u+0.0001);
n2 = sqrt(v.*v+0.0001);
a = norm - s1.*s2;
loss = a'*([n1+n2;0]);

partialY = ewx.*pplus.*pplus;
regStdY2 = 1/l2/stdY2*(Y2 - barY2);
partialstdY2 = 2*X2*(partialY'.*(regStdY2 - sum(regStdY2)/l2));

reg = -s1.*([n1+n2;0]);
g11 = 2*X2*(partialY/l2.*(reg'*X2-reg'*barX2))';

g1 = (g11*stdY2 - reg'*s2*stdY2*partialstdY2)/stdY2/stdY2;
g2 = [u./n1; 0; v./n2; 0];

g1 = [g1; g1];
gradient = [a; a].*g2 + g1;


