if ~exist('X', 'var') || ~exist('Xf', 'var') 
X = sprand(1e4,1e3,0.03);
Xf = full(X);
end
 
Xfs = spshuffle(X,ks);

% [M,N] = size(X);
% [i,j,s] = find(X);
% ii = cell(1,N);
% for jj = 1:N
%     ii{jj} = i(j == jj);
% end