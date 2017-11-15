function y=simulateSoftData(Element,MID_list,mu,SIGMA)
% mu is l by d matrix. l=the number of labels; d=the number of features
% sigma is d by d by l matrix

n=length(Element.Color);
d=size(mu,2);
k=size(mu,1);
y=zeros(n,d);
p=zeros(1,k);
for i=1:n
    MID=MID_list(i);
    y(i,:)=mvnrnd(mu(MID,:),SIGMA(:,:,MID));
    p(MID)=p(MID)+1;
end

%GMModel = gmdistribution(mu,SIGMA,p/n);
%figure;
%ezcontour(@(x,y)pdf(GMModel,[x y]),[min(y(:,1)) max(y(:,1))],[min(y(:,2)) max(y(:,2))]);
end