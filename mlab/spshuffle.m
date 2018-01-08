function X = spshuffle(X, ks)
ks = ks(:)';
K_vals = unique(ks);
[~,N] = size(X);

if issparse(X)
    [ii,jj,ss] = sp2cell(X);
    ii = cellshuf(ii, ks, K_vals);
    X = cell2sp(ii,jj,ss);
else
    for k = K_vals
        is = find(ks == k);
        for jj = 1:N
            X(is, jj) = X(is(randperm(length(is))),jj);
        end
    end
end
end

function ii = cellshuf(ii, ks, K_vals)
for k = K_vals
    is_in_subset = ks == k;
    num_in_class = sum(is_in_subset);
    for jj = 1:length(ii)
        i_mask = is_in_subset(ii{jj});
        num_to_shuf = sum(i_mask);
        ii{jj}(i_mask) = randperm(num_in_class, num_to_shuf);
    end
end
end

function [ii,jj,ss] = sp2cell(X)
[~,N] = size(X);
[i,j,s] = find(X);
Ns = zeros(N,1);
for j_ind = 1:N
    Ns(j_ind) = sum(j == j_ind);
end
ii = mat2cell(i, Ns);
jj = mat2cell(j, Ns);
ss = mat2cell(s, Ns);
end

function X = cell2sp(ii,jj,ss)
i = cell2mat(ii);
j = cell2mat(jj);
s = cell2mat(ss);
X = sparse(i,j,s);
end