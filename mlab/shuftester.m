clear

M = 1e4; N = 1e3; K = 20; s = 0.03;
ks = randi(K, M, 1);
%X = sparse(ks + rand(M,N)*0.1);
X = sprand(M, N, s);

featmask = logical(sparse(N,1));
%%
tic
g = shufgen(X,ks);
Xs = cell(1,100);
for i = 1:100
    featmask(i*10) = true;
    Xs{i} = g(featmask);
    featmask(i*10) = false;
end
toc

%tic
%Xfs = spshuffle(Xf,ks);
%toc

% [M,N] = size(X);
% [i,j,s] = find(X);
% ii = cell(1,N);
% for jj = 1:N
%     ii{jj} = i(j == jj);
% end