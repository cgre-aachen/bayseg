function Element=Kriging(Element,knownInfo,GaussianCov,CorrelationLength)


N=length(Element);
S0=zeros(N,3);
for i=1:N
    S0(i,:)=Element(i).Center;
end
S=knownInfo(:,1:3);
Z=zeros(size(S,1),3);
for k=1:size(S,1)
  Z(k,:)=knownInfo(k,4:6)/norm(knownInfo(k,4:6));
end


Z_hat=OrdinaryKriging(S0,S,Z,GaussianCov,CorrelationLength);
for i=1:N
    Element(i).Orientation=Z_hat(i,:)/norm(Z_hat(i,:));
end

%===for plot=======================================================================
u=Z(:,1);
v=Z(:,2);
w=Z(:,3);
%
u_hat=Z_hat(:,1);
v_hat=Z_hat(:,2);
w_hat=Z_hat(:,3);

%===for plot=======================================================================
hold on;
MID_list=zeros(N,1);
PlotFig3(Element,MID_list,0,0,10,[0 1 2 3]);

quiver3(S(:,1),S(:,2),S(:,3),u,v,w,0,'Color',[0 0 1]);

quiver3(S0(1:100:N,1),S0(1:100:N,2),S0(1:100:N,3),u_hat(1:100:N),v_hat(1:100:N),w_hat(1:100:N),0,'Color',[1 0 0]);
view(3);
box on;
axis equal;
hold off;
%==========================================================================
end
    