function WriteGmsh(msh_file_name,data_file_name,MC_inferred,Prob,InfEntropy,Mset,Element,space,flag)
% flag = 1 by 4 logistic vector indicating yes or no
% flag(1): GMM result;
% flag(2): HMRF results;
% flag(2): Prob;
% flag(4): entropy;

copyfile(msh_file_name,data_file_name);

num_of_element = length(Element);
EleID = zeros(1,num_of_element);
for i=1:num_of_element
    EleID(i) = Element(i).Num;
end

% open the msh file with permision to append
fid = fopen(data_file_name,'a');

if flag(1)==1
    fprintf(fid,'$ElementData\n');
    fprintf(fid,'1\n');
    fprintf(fid,'"GMM"\n');
    fprintf(fid,'1\n');
    fprintf(fid,'1\n');
    fprintf(fid,'3\n');
    fprintf(fid,'0\n');
    fprintf(fid,'1\n');
    fprintf(fid,'%d\n',num_of_element);
    D=[EleID; (MC_inferred(:,1))'];
    fprintf(fid,'%d %d\n',D);
    fprintf(fid,'$EndElementData\n');
end

if flag(2)==1
    chain_length = size(MC_inferred,2);
    max_idx=chain_length/space;
    for i=1:max_idx
        fprintf(fid,'$ElementData\n');
        fprintf(fid,'1\n');
        fprintf(fid,'"HMRF Iteration"\n');
        fprintf(fid,'1\n');
        fprintf(fid,'%d\n',i);
        fprintf(fid,'3\n');
        fprintf(fid,'%d\n',i-1);
        fprintf(fid,'1\n');
        fprintf(fid,'%d\n',num_of_element);
        D=[EleID; (MC_inferred(:,i*space))'];
        fprintf(fid,'%d %d\n',D);
        fprintf(fid,'$EndElementData\n');
    end
end

if flag(3)==1
    for i=1:length(Mset)
        fprintf(fid,'$ElementData\n');
        fprintf(fid,'1\n');
        fprintf(fid,'"Prob %d"\n',i);
        fprintf(fid,'1\n');
        fprintf(fid,'1\n');
        fprintf(fid,'3\n');
        fprintf(fid,'0\n');
        fprintf(fid,'1\n');
        fprintf(fid,'%d\n',num_of_element);
        D=[EleID; (Prob(:,i))'];
        fprintf(fid,'%d %d\n',D);
        fprintf(fid,'$EndElementData\n');
    end
end

if flag(4)==1
    fprintf(fid,'$ElementData\n');
    fprintf(fid,'1\n');
    fprintf(fid,'"Entropy"\n');
    fprintf(fid,'1\n');
    fprintf(fid,'1\n');
    fprintf(fid,'3\n');
    fprintf(fid,'0\n');
    fprintf(fid,'1\n');
    fprintf(fid,'%d\n',num_of_element);
    D=[EleID; InfEntropy'];
    fprintf(fid,'%d %d\n',D);
    fprintf(fid,'$EndElementData\n');
end

fclose(fid);
end

