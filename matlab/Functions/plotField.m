function [region,x,y,z] = plotField(Element,value_list,cmap)
% plot the 2D segmentation results

grid_matrix = Element.Grid;
center_matrix = Element.Center;

flag = ~isnan(value_list);
x_min = min(center_matrix(flag,1));
x_max = max(center_matrix(flag,1));
y_min = min(center_matrix(flag,2));
y_max = max(center_matrix(flag,2));

x = unique(center_matrix(:,1));
y = unique(center_matrix(:,2));
z = unique(center_matrix(:,3));

size_grid = max(grid_matrix);
if size_grid(3) == 1    
    coord = grid_matrix(:,[2,1]);    
    temp = accumarray(coord,value_list,[],[],NaN);    
    h = pcolor(x,y,temp);    
    h.EdgeColor = 'none';
    xlim([x_min,x_max]);
    ylim([y_min,y_max]);    
    %axis image;
    axis equal;    
    colormap(cmap);    
    colorbar;
    region = flip(temp);
end