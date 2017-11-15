function [Vol,C] = PolyhedronVolume(p)

T = delaunayTriangulation(p);
t = T.ConnectivityList;
e1 = p(t(:,2),:)-p(t(:,1),:);
e2 = p(t(:,3),:)-p(t(:,1),:);
e3 = p(t(:,4),:)-p(t(:,1),:);
V = abs(dot(cross(e1,e2,2),e3,2))/6;
Vol = sum(V);
%format long;
%display(V);
idx=find(V>1e-10);
%display(idx);

CC = circumcenter(T);

%display(V(idx));
%display(CC(idx,:));
C = V(idx)'*CC(idx,:)/sum(V);
end
