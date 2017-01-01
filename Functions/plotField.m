function [region,x,y,z] = plotField(Element,value_list)
% plot the 2D segmentation results
%figure;
grid_matrix = Element.Grid;
center_matrix = Element.Center;
x = unique(center_matrix(:,1));
y = unique(center_matrix(:,2));
z = unique(center_matrix(:,3));
size_grid = max(grid_matrix);
if size_grid(3) == 1
    coord = [size_grid(2)-(grid_matrix(:,2)-1),grid_matrix(:,1)];
    notNaN = ~isnan(value_list);
    region = accumarray(coord(notNaN,:),value_list(notNaN),[],[],NaN);
    imagesc(x([1,length(x)]),flip(y([1,length(y)])),region);    
    set(gca,'Ydir','normal');
    axis image;
    axis equal;
    cm = jet;
    nan_clr = [1 1 1];
    %# find minimum and maximum
    value_min=min(value_list) - 1e-7;
    value_max=max(value_list) + 1e-7;
    %# size of colormap
    n = size(cm,1);
    %# color step
    clr_step=(value_max-value_min)/n;
    colormap([nan_clr; cm]);
    caxis([value_min-clr_step value_max]);
    colorbar;
end