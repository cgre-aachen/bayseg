function [Node,Element]=ReadGmsh(Gmsh_file_name,dimension,resolution,order)
% This function is used for reading a Gmsh file and returns the Node structure and
% Element structure;
% Gmsh_file_name is the Gmsh file to be read;
% dimension is the dimension of this Gmsh file: 2 or 3;
% resolution is a 1 by 2 (or 1 by 3 for 3D) vector of the grid size of each dimension.

if dimension == 2
    target_ele_type = 3; % 4-node quadrangle
    num_of_nodes_perElement = 4;    
end

if dimension == 3
    target_ele_type = 5; % 8-node hexahedron
    num_of_nodes_perElement = 8;
end

Node=struct('Coord',[0 0 0],'EleList',0);
Element=struct('Num',0,'Grid',zeros(1,3),'Nodes',zeros(1,num_of_nodes_perElement),'CoordMatrix',zeros(num_of_nodes_perElement,3),'Center',zeros(1,3),'Degree',0,'Color',0,'SelfU',[],'Neighbors',0);
%============================read file=====================================
fid=fopen(Gmsh_file_name,'r');
while ~feof(fid)
    this_line=fgetl(fid);
    %=====================================================================
    % If there is no keyword in this_line, jump to next line
    while strcmp(this_line,'$Nodes')==0 && strcmp(this_line,'$Elements')==0
        this_line=fgetl(fid);
    end
    %====================================================================
    % If the keyword is "$Nodes"
    if strcmp(this_line,'$Nodes')==1
        this_line=fgetl(fid);
        i=1;
        while ~isempty(this_line) && ~feof(fid) && strcmp(this_line,'$EndNodes')~=1
            temp=cell2mat(textscan(this_line,'%f'));
            if length(temp)>1   % jump the line: # of nodes
                Node(i).Coord=temp(2:4);
                Node(i).EleList=zeros(8,1);
                this_line=fgetl(fid);
                fprintf('Construct nodes... Node_Num=%d \n',i);
                i=i+1;
            else
                this_line=fgetl(fid);
            end
        end
        this_line=fgetl(fid); % jump the line '$EndNodes'
    end
    %======================================================================
    % If the keyword is "$Elements"
    if strcmp(this_line,'$Elements')==1
       this_line=fgetl(fid);
       i=1;
       while ~isempty(this_line) && ~feof(fid) && strcmp(this_line,'$EndElements')~=1
           temp=cell2mat(textscan(this_line,'%f'));
           if length(temp)>1 % jump the line: # of elements
               ele_type = temp(2);               
               if ele_type == target_ele_type
                   Element(i).Num = temp(1);
                   Element(i).Nodes = temp(4+temp(3):length(temp));
                   for k=1:num_of_nodes_perElement
                       Element(i).CoordMatrix(k,:)=Node(Element(i).Nodes(k)).Coord;
                       idx=nnz(Node(Element(i).Nodes(k)).EleList)+1;
                       Node(Element(i).Nodes(k)).EleList(idx)=i;
                   end
                   if dimension == 3
                       [~,Element(i).Center]=PolyhedronVolume(Element(i).CoordMatrix);
                       Element(i).Grid = round((Element(i).Center+0.5*resolution)./resolution);
                   end
                   if dimension == 2
                       [geom,~,~] = polygeom(Element(i).CoordMatrix(:,1),Element(i).CoordMatrix(:,2));
                       Element(i).Center = [geom(2:3) 0];
                       Element(i).Grid = [round((Element(i).Center(1:2)+0.5*resolution)./resolution) 1];
                   end                   
                   Element(i).Degree=0;
                   Element(i).Color=0;
                   this_line=fgetl(fid);
                   fprintf('Construct elements... Element_Num=%d \n',i);
                   i=i+1;
               else
                   this_line=fgetl(fid);
               end               
           else
                this_line=fgetl(fid);
           end
       end       
    end
end
fclose(fid);
%=================finish reading file======================================

%=========================================================================
% Find the neighbors
grid_matrix = vec2mat([Element.Grid],3);
parfor i=1:length(Element)
    temp=[];
    for j = 1:dimension
        idx_j = find(abs(grid_matrix(:,j)-Element(i).Grid(j))<=order);
        if j == 1
            temp = idx_j;
        else
            temp = intersect(temp,idx_j);
        end
    end
    NeighborsID=setdiff(temp,[i;0]);
    Element(i).Degree=length(NeighborsID);
    Element(i).Neighbors=NeighborsID; 
    fprintf('Finding neighbors... Element_Num=%d \n',i);    
end
%==========================================================================            
fprintf('coloring... \n');
[Element,m]=LRSC(Element);
fprintf('The total number of colors is=%d \n',m);
end 