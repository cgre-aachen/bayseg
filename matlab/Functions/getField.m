function [R_field,ux2,uy2,loc] = getField(Field,ux1,uy1,loc)
Field_data = formatField(ux1,uy1,Field);

if isempty(loc)
    figure('units','normalized','outerposition',[0 0 1 1]);
    imagescwithnan(ux1,uy1,Field,viridis,[1 1 1]);
    %caxis([0 5]);
    title('Make a polygon around field of interest by using left mousebutton, when finished use other.');    
    but = 1;
    loc=zeros(50,2);
    count = 0;
    while but == 1
        hold on
        % use left mousebutton to choose and right button to leave the loop
        [locX,locY,but] = ginput(1) ;
        if but == 1;
            count = count +1;
            plot(locX,locY,'kx','MarkerSize',12)
            loc(count,:)=[locX,locY];
            hold on
        else  break;
        end
    end
    loc = loc(1:count,:);
end
    

xfilt = loc(:,1) ;
yfilt = loc(:,2) ;

c = convhull(xfilt,yfilt);
in = inpolygon(Field_data(:,1),Field_data(:,2),xfilt(c),yfilt(c)) ;

Field_data_selected = Field_data(in,:) ;

[ux2,~,bx2] = unique(Field_data_selected(:,1));
[uy2,~,by2] = unique(Field_data_selected(:,2));
R_field = accumarray( [max(by2)-(by2-1),bx2], Field_data_selected(:,3),[],[], NaN);
%R_field = accumarray( [by2,bx2], Field_data_selected(:,3),[],[], NaN);
end