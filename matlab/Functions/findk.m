function [flag,k]=findk(Element,v,m)
num_of_neighbors = length(Element.Neighbors{v});
nc(num_of_neighbors,1) = 0;
for j=1:num_of_neighbors
    if Element.Color(Element.Neighbors{v}(j))>0
        nc(j)=Element.Color(Element.Neighbors{v}(j));
    end
end
u=-1;
k=0;
while k<=m && ~isempty(u)
    k=k+1;
    u=intersect(k,nc);
end
if k<=m
    flag=1;
else
    flag=0;
end
end