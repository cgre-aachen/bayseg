function logpdf = invwishpdf(X,S,v) 
p = size(S,1);
logpdf = v/2*log(det(S))-(v+p+1)/2*log(det(X))-0.5*trace(S/X)-(v*p/2*log(2)+logMvGamma(v/2,p));
end