function s = ThorntonSeparaIdx(X,L,ClassID)
% Return the Thornton's separability index
% X is the n by d matrix with n entries of d-dimension vectors;
% L is the n by 1 list of all labels corresponding to the matrix X;
% ClassID is the list of all possible labels;

N = length(L);

for i = 1:length(ClassID)
	temp = zeros(N,1);
	temp(L == ClassID(i)) = 1;
	m = zeros(N,1); % the idx of Elements whos nearest neighbor has the same label: 1 ----- same , 0 ---- different
	parfor j = 1:N
	    D = pdist2(X,X(j,:));
        [~,I] = sort(D);
        if temp(I(1)) == temp(I(2))
        	m(j) = 1;
        end
    end
    s(i) = sum(m)/N;
end

poolobj=gcp('nocreate');
delete(poolobj);

end