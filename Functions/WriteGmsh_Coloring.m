function WriteGmsh_Coloring(msh_file_name,data_file_name,Element)
copyfile(msh_file_name,data_file_name);
num_of_element = length(Element);
EleID = zeros(1,num_of_element);
C = zeros(1,num_of_element);
for i=1:num_of_element
    EleID(i) = Element(i).Num;
    C(i) = Element(i).Color;
end

fid = fopen(data_file_name,'a');
fprintf(fid,'$ElementData\n');
fprintf(fid,'1\n');
fprintf(fid,'"coloring"\n');
fprintf(fid,'1\n');
fprintf(fid,'1\n');
fprintf(fid,'3\n');
fprintf(fid,'1\n');
fprintf(fid,'1\n');
fprintf(fid,'%d\n',num_of_element);
D=[EleID; C];
fprintf(fid,'%d %d\n',D);
fprintf(fid,'$EndElementData\n');
end