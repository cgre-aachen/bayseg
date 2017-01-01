function Element_updated=CalculateBeta(Element,orientation,flag)
if strcmp(flag,'aniso')
    Element_updated=Element;    
    for i=1:length(Element)
        n_neigh = length(Element(i).Neighbors);
        Element_updated(i).Beta=zeros(n_neigh,1);
        for j=1:n_neigh
            lij=(Element(Element(i).Neighbors(j)).Center-Element(i).Center);
            lij=lij/norm(lij);
            orientation=orientation/norm(orientation);
            lji=-lij;
            Element_updated(i).Beta(j)=2-abs(lij*orientation')-abs(lji*orientation');        
        end
    end
end
end
