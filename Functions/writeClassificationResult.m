function ID=writeClassificationResult(Element,MC,Data)

N=length(Element);
Chain_length=size(MC,2);
ID(N,Chain_length)=0;
%=============================================

P(N,3)=0;
V=MC;

for i=1:N
    P(i,:)=Element(i).Center;
end

% construct the query points
Xq(:,1)=Data(:,1);
Yq(:,1)=Data(:,2);
Zq(:,1)=Data(:,3);


for i=1:Chain_length
    F=scatteredInterpolant(P,V(:,i),'nearest','nearest');
    ID(:,i)=F(Xq,Yq,Zq);
end

end