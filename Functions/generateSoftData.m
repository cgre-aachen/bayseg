function [y,MC]=generateSoftData(Element,mu,SIGMA)
% this function is only used for the 3_layers model
% mu is l by d matrix. l=the number of labels; d=the number of features
% sigma is d by d by l matrix

    function ind1=surf1(coord)
        ind1=0.5*coord(1)-coord(3);       
    end

    function ind2=surf2(coord)
        ind2=0.5*coord(1)-coord(3)+10;
    end
n=length(Element);
d=size(mu,2);
y=zeros(n,d);
p=zeros(1,3);
MC=zeros(n,1);
for i=1:n
    coord=Element(i).Center;
    ind1=surf1(coord);
    ind2=surf2(coord);
    if ind1<0 && ind2<0
        MC(i)=1;
        y(i,:)=mvnrnd(mu(1,:),SIGMA(:,:,1));
        p(1)=p(1)+1;
    else if ind1<=0 && ind2>=0
            MC(i)=2;
            y(i,:)=mvnrnd(mu(2,:),SIGMA(:,:,2));
            p(2)=p(2)+1;
        else
            MC(i)=3;
            y(i,:)=mvnrnd(mu(3,:),SIGMA(:,:,3));
            p(3)=p(3)+1;
        end
    end
end

GMModel = gmdistribution(mu,SIGMA,p/n);
ezcontour(@(x,y)pdf(GMModel,[x y]),[min(y(:,1)) max(y(:,1))],[min(y(:,2)) max(y(:,2))]);

WriteGmsh_y('myscriptFine.msh','myscriptFine_y_3Layers.msh',y,Element);

end