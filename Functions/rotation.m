function Rotation_Matrix = rotation(x,y,theta)
% x,y: two base vectors defining a plane, theta is the rotation angle in this plane 
u=x/norm(x);  % norm is just euclidean distance
v=y-u'*y*u;
v=v/norm(v);
Rotation_Matrix = eye(length(x))-u*u'-v*v'+[u v]*[cos(theta) -sin(theta);sin(theta) cos(theta)]*[u v]';
end