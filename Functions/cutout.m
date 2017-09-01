function dataList = cutout(coord,dataList,loc)
% This function is used for deleting some pixels within several polugons 
% defined by a cell "loc"

x = coord(:,1);
y = coord(:,2);

for i = 1:length(loc)
    temp = loc{i};
    xfilt = temp(:,1);
    yfilt = temp(:,2);
    in = inpolygon(x,y,xfilt,yfilt);
    dataList(in,:) = NaN;
end