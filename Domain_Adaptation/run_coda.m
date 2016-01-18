

domains=cell(4,1);
domains{1}='books';
domains{2}='dvd';
domains{3}='electronics';
domains{4}='kitchen';

pos = 10;
neg = 10;
gamma = 0.001;
epsilon = 1.0;
decreRatio=1.05;

for i=1:size(domains,1)
        target=domains{i};
        for j = 1:size(domains,1)
		source=domains{j};
		if i == j
			continue;
		end
		fprintf('====================== source %s -> target %s ====================\n', source, target);
		load(['./data/', source, '.', target, '.X.mat']);
		X = [X; ones(1,size(X,2))];
		Y = csvread(['./data/', source, '.', target, '.Y.dat']);
		Y = Y';
                idxL = 1:2000;
                idxU = 4001:length(Y);
                idxT=idxU;
		idxSS = idxL;
		idxTT = idxU;
		[W, loss, acc] = coda(X, Y, idxL, idxU, idxT, pos, neg, 0, 100, epsilon, idxSS, idxTT, gamma,decreRatio);
		save(['./results/', source, '.', target, '.mat'], 'W', 'acc');
        end
end


