function [region,x,y,z] = plotField(Element,value_list,cmap)
% plot the 2D segmentation results
%figure;
grid_matrix = Element.Grid;
center_matrix = Element.Center;
x = unique(center_matrix(:,1));
y = unique(center_matrix(:,2));
z = unique(center_matrix(:,3));
size_grid = max(grid_matrix);
if size_grid(3) == 1    
    coord = grid_matrix(:,[2,1]);    
    region = accumarray(coord,value_list,[],[],NaN);    
    h = pcolor(x,y,region);
    h.EdgeColor = 'none';
    set(gca,'Ydir','normal');
    axis image;
    axis equal;    
    colormap(cmap);    
    colorbar;
end