function WriteGmsh_y(msh_file_name,data_file_name,data,Element)
% write Gmsh file to visualize soft data in 3D meshed model
%data is a n by d matrix, n elements and d dimentions

copyfile(msh_file_name,data_file_name);

dimention = size(data,2);
num_of_element = length(Element);
EleID = zeros(1,num_of_element);
for i=1:num_of_element
    EleID(i) = Element(i).Num;
end

% open the msh file with permision to append
fid = fopen(data_file_name,'a');

for i=1:dimention    
    fprintf(fid,'$ElementData\n');
    fprintf(fid,'1\n'); % 1 string tag
    fprintf(fid,'"Feature %d"\n',i); % the name of the view
    fprintf(fid,'1\n'); % 1 real tag
    fprintf(fid,'%d\n',i); % the time value
    fprintf(fid,'3\n'); % 3 integer tags
    fprintf(fid,'%d\n',i-1); % the time step
    fprintf(fid,'1\n'); % 1-component (scale field)
    fprintf(fid,'%d\n',num_of_element); % number of values
    D=[EleID; (data(:,i))'];
    fprintf(fid,'%d %d\n',D);
    fprintf(fid,'$EndElementData\n');
end

fclose(fid);
end