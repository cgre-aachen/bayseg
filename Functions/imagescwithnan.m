function [h] = imagescwithnan(ux,uy,a,cm,nanclr)
% IMAGESC with NaNs assigning a specific color to NaNs

%# find minimum and maximum
amin=min(a(:));
amax=max(a(:));
%# size of colormap
n = size(cm,1);
%# color step
dmap=(amax-amin)/n;

%# standard imagesc
him = imagesc(ux([1,length(ux)]),flip(uy([1,length(uy)])),a);
set(gca,'Ydir','normal');
axis image;
axis equal;
%# add nan color to colormap
colormap([nanclr; cm]);
%# changing color limits
caxis([amin-dmap-1e-5 amax]);
%# place a colorbar
%hcb = colorbar;
%# change Y limit for colorbar to avoid showing NaN color
%ylim(hcb,[amin amax])

if nargout > 0
    h = him;
end