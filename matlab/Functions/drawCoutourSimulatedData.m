function drawCoutourSimulatedData(y,MID_list,mu,SIGMA)
k=size(mu,1);
p=zeros(1,k);
n=length(MID_list);
for i=1:n
    MID=MID_list(i);    
    p(MID)=p(MID)+1;
end

GMModel = gmdistribution(mu,SIGMA,p/n);
figure;
ezcontour(@(x,y)pdf(GMModel,[x y]),[min(y(:,1)) max(y(:,1))],[min(y(:,2)) max(y(:,2))]);
end
    