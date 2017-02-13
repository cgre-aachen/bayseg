function loc = getPolygon(A,ux,uy)

q = false;
loc = {};

figure('units','normalized','outerposition',[0 0 1 1]);
imagescwithnan(ux,uy,A,viridis,[1 1 1]);

while ~q
    title('Make a polygon around field of interest by using left mousebutton, when finished use other.');
    but = 1;    
    temp = [];
    while but == 1
        hold on
        % use left mousebutton to choose and right button to leave the loop
        [locX,locY,but] = ginput(1) ;
        if but == 1            
            plot(locX,locY,'kx','MarkerSize',12)
            temp=cat(1,temp,[locX,locY]);
            hold on
        else
            polygon = [temp;temp(1,:)];
            plot(polygon(:,1),polygon(:,2),'r-');
            break;
        end
    end
    loc = cat(1,loc,{temp});
    prompt = 'Do you want more polygons? y/n [y]: ';
    str = input(prompt,'s');
    if isempty(str)
        str = 'y';
    end
    if str == 'n'
        q = true;
    end        
end
end
    
    

