function [theta,phi]=vecter2angle(X)
theta=acos(X(3)/sqrt(X(1)^2+X(2)^2+X(3)^2));

if X(1)>0 && X(2)>0
    phi=atan(X(2)/X(1));
end
if X(1)<0 && X(2)>0
    phi=atan(X(2)/X(1))+pi;
end
if X(1)<0 && X(2)<0
    phi=atan(X(2)/X(1))+pi;
end
if X(1)>0 && X(2)<0
    phi=atan(X(2)/X(1))+2*pi;
end
if X(2)>0 && X(1)==0
    phi=pi/2;
end
if X(2)<0 && X(1)==0
    phi=3*pi/2;
end
if X(2)==0 && X(1)>0
    phi=0;
end
if X(2)==0 && X(1)<0
    phi=pi;
end
if X(1)==0 && X(2)==0
    phi=0;
end

end