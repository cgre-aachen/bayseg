function Element = detectNeighborDirection(Element,dimension)
% this function is used for detecting the directions of the neighbors to
% the center element.
% Element.Direction is n_neigh by n_direction logic vector.
n_direction = max([Element.Degree])/2;
num_of_ele = length(Element.Color);

if dimension == 1
    base = [1; 
            0; 
            0];
end

if dimension == 2
    base = [1 0 1 -1; 
            0 1 1 1; 
            0 0 0 0];
end
if dimension == 3
    base = [1 0 0 1 1 0 0 1 -1 1 -1 1 -1;
            0 1 0 1 -1 1 -1 0 0 1 1 -1 -1;
            0 0 1 0 0 1 1 1 1 1 1 1 1]; 
end

Direc = cell(num_of_ele,1);
Nei = Element.Neighbors;
Grid = Element.Grid;
parfor i=1:num_of_ele
    n_neigh = length(Nei{i});
    Direc{i} = zeros(n_neigh,1);
    for j=1:n_neigh
        lij= Grid(Nei{i}(j),:)-Grid(i,:);
        if dimension == 1
            lij = base';
        end
        if dimension == 2
            if (lij(2) < 0) || (lij(2) == 0 && lij(1) < 0)
                lij = -1*lij;
            end            
        end
        if dimension == 3
            if (lij(3) < 0) || (lij(3) == 0 && (lij(2) < 0) || (lij(2) == 0 && lij(1) < 0))
                lij = -1*lij;
            end            
        end
        [~,flag] = min(sum((lij'*ones(1,n_direction)-base(:,1:n_direction)).^2));
        Direc{i}(j) = flag;
    end    
end
Element.Direction = Direc;
end