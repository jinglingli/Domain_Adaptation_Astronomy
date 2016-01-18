
function [varA, varB, w, l, converged] = feature_split(w_last, xTrA, yTrA, beliefA, xTrB, yTrB, beliefB,  maxIter, lambda, xU, epsilon, xSS, ySS, xTT, stdXSS, stdXTT, gamma)


d = size(xTrA, 1);
alpha = 0;
c = 0.01;
alpha2 = 0;
c2 = 0.01;
schedule = 1.2;
converged = false;
w1 = w_last(1:d-1);
w2 = w_last(d+1:end-1);
old_overlap = sum(w1.*w1.*w2.*w2);
old_expansion = balcanExpansion(w_last, xU, epsilon);
iter = 0;
while c < 100
	iter = iter+1;
	[w,loss]= minimize(w_last, @computeLossGrad, maxIter, xTrA, yTrA,  beliefA, xTrB, yTrB, beliefB, lambda, xU, epsilon, xSS, ySS, xTT, stdXSS, stdXTT, gamma, alpha, c, alpha2, c2);
	w1 = w(1:d-1);
	w2 = w(d+1:end-1);
    if(w==w_last)
        fprintf('True \n');
    else
        fprintf('False \n');
    end
        overlap = sum(w1.*w1.*w2.*w2);
	alpha = alpha + c*overlap;
	if overlap >= 0.25*old_overlap
		c = c*schedule;
	end

        expansion = balcanExpansion(w, xU, epsilon);
        alpha2 = alpha2 + c2*expansion;
        if expansion >= 0.25*old_expansion
        	c2 = c2*schedule;
        end
	fprintf('overlap %f, expansion %f\n', overlap, expansion);
	if overlap < 1e-1 && expansion < 1e-1
		converged = true;
		fprintf('converged with c = %f, alpha = %f, c2 = %f, alpha2 = %f, at iteration %d\n', c, alpha, c2, alpha2, iter);
		break;
    end
	w_last = w;
    %fprintf('wlast %f, w %f \n', w_last, w);
	old_overlap = overlap;
	old_expansion = expansion;
end
l = loss(end);
w1 = w(1:d-1);
w2 = w(d+1:end-1);

fprintf('maximum w1 = %f, maximum w2 = %f\n', max(abs(w1)), max(abs(w2)));
%varA = find(abs(w1)>abs(w2));
%varB = find(abs(w2)>abs(w1));
%shared = find(abs(w1)==abs(w2));
varA = find(abs(w1) > 0.1 & abs(w1) > abs(w2));
varB = find(abs(w2) > 0.1 & abs(w2) > abs(w1));
shared = find(abs(w1) > 0.1 & abs(w1) == abs(w2));
temp = randsample(length(shared), floor(length(shared)/2));
varA = [varA;shared(temp)];
shared(temp)=0;
varB = [varB;shared(shared>0)];
fprintf('classifier A selectes %d, classifier B selectes %d\n', length(varA), length(varB));
