function Element = constructElements(x,y,z,order)
% This function is used for construct Element structure given the domain
% size x,y,z and the dimensions of the problem

grid_matrix = (combvec(1:length(x),1:length(y),1:length(z)))';
num_of_elements = size(grid_matrix,1);

Element=struct('Grid',[],'Center',[],'Degree',[], ... 
    'Color',[],'Neighbors',cell(1,1));

Element.Grid = grid_matrix;
Element.Center = [x(grid_matrix(:,1)),y(grid_matrix(:,2)),z(grid_matrix(:,3))];

D(num_of_elements,1) = 0;
C(num_of_elements,1) = 0;
Nei = cell(num_of_elements,1);
fprintf('Finding neighbors...\n');    
parfor i=1:num_of_elements
    idx_matrix = abs(grid_matrix-ones(num_of_elements,1)*grid_matrix(i,:)) <= order;
    temp = find(sum(idx_matrix,2)==3);
    NeighborsID=setdiff(temp,[i;0]);
    D(i)=length(NeighborsID);
    C(i) = 0;
    Nei{i}=NeighborsID;  % column vector;    
end
Element.Degree = D;
Element.Color = C;
Element.Neighbors = Nei;

%==========================================================================            
fprintf('coloring... \n');
[Element,m]=LRSC(Element);
fprintf('The total number of colors is=%d \n',m);
end 