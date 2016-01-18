function [loss,gradient]= balcanExpansion(w, xU, epsilon)

% INPUT:
% w    2*d weight vector
% xU   dxn    unlabeled inputs
% epsilon  hyperparameter in balcan's epsilon expansion theory (Balcan et. al, 2004)

[d, n] = size(xU);

w1 = w(1:d);
w2 = w(d+1:end);

v1 = w1'*xU;
v2 = w2'*xU;

y1 = sign(v1);
y2 = sign(v2);

ewx1 = exp(-v1.*y1);
ewx2 = exp(-v2.*y2);

tau = 0.8;
p1 = 1./(1+ewx1);
p2 = 1./(1+ewx2);

c1 = 1./(1+exp(-50*(p1-tau)));
c2 = 1./(1+exp(-50*(p2-tau)));

loss = epsilon*min([c1*c2', (1-c1)*(1-c2)']) - c1*(1-c2)' - (1-c1)*c2';
loss = loss/n;

if loss <= 0
        loss = 0;
        gradient = zeros(2*d, 1);
else
	pp1 = c1.*c1.*exp(-50*(p1-tau))*50;
	pp2 = c2.*c2.*exp(-50*(p2-tau))*50;

    	reg1 = ewx1.*y1.*p1.*p1;
    	reg2 = ewx2.*y2.*p2.*p2;
        if c1*c2' < (1-c1)*(1-c2)'
            pre1 = epsilon.*c2 - (1-c2) + c2;
            pre2 = epsilon.*c1 + c1 - (1-c1);
        else
            pre1 = -epsilon.*(1-c2) - (1-c2) + c2;
            pre2 = -epsilon.*(1-c1) + c1 - (1-c1);
        end
	gradient = [ xU*(pre1.*reg1.*pp1)'; xU*(pre2.*reg2.*pp2)']/n;
end


