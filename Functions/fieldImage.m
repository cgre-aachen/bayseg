function [h] = fieldImage(ux,uy,M,cmap)
% IMAGESC with NaNs assigning a white color to NaNs

him = pcolor(ux,uy,flip(M));
him.EdgeColor = 'none';
set(gca,'Ydir','normal');
axis image;
axis equal;
colormap(cmap);
colorbar;

if nargout > 0
    h = him;
end