function field_fine = changeResolution(field_coarse,x_coarse,y_coarse,x_fine,y_fine)
[X,Y] = meshgrid(x_coarse,y_coarse);
[Xq,Yq] = meshgrid(x_fine,y_fine);
field_fine = interp2(X,flip(Y),field_coarse,Xq,flip(Yq),'nearest');
%field_fine = interp2(X,Y,field_coarse,Xq,Yq,'nearest');
% figure;
% imagescwithnan(x_fine,y_fine,field_fine,viridis,[1 1 1]);
end