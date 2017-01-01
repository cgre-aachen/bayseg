function C=GaussianCov(X1,X2,CovaranceLength)
r=sqrt(sum((X1-X2).^2));
C=exp(-(r/CovaranceLength)^2);
end
