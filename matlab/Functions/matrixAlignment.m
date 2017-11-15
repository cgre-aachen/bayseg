function X_updated = matrixAlignment(X)
% this function is used to make the multiple matrices all having field values
% at the same pixels, if there are NaNs at some pixels of a certain matrix,
% the pixels of other matrices will also change their values to NaN
% X is a 3D matrix containing multiple layers, each layer is a data set.

flag = isnan(sum(X,3));
layers_of_X = size(X,3);
X_updated = X;
for i = 1:layers_of_X
    temp = X(:,:,i);
    temp(flag) = NaN;
    X_updated(:,:,i) = temp;
end
end