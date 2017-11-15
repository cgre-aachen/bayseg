function y=interpoFromObservations(Element,Data)

N=length(Element);
num_of_features=size(Data,2)-3;
y(N,num_of_features)=0;
%=============================================

P=Data(:,1:3);
V=Data(:,4:size(Data,2));

% construct the query points
Xq(N,1)=0;
Yq(N,1)=0;
Zq(N,1)=0;
for i=1:N
    Xq(i)=Element(i).Center(1);
    Yq(i)=Element(i).Center(2);
    Zq(i)=Element(i).Center(3);
end

for i=1:num_of_features
    F=scatteredInterpolant(P,V(:,i),'nearest','nearest');
    y(:,i)=F(Xq,Yq,Zq);
end

end
    