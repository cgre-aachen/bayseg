function y = retrieve(Element,Field)
% this function is used for retrieving data list from a 2D or 3D image
% "field" according to the pixel order in the structure "Element"

grid_matrix = Element.Grid;
size_grid = max(grid_matrix);
y = NaN(size(grid_matrix,1),1);
if size_grid(3) == 1
    parfor i = 1:length(y)
        coord = [size_grid(2)-(grid_matrix(i,2)-1),grid_matrix(i,1)];        
        y(i) = Field(coord(1),coord(2));
    end
end
