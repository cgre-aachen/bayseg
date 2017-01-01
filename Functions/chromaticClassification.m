function [para_scanorder,num_of_color]=chromaticClassification(Element)

C = Element.Color;
num_of_color=max(C);
para_scanorder=cell(num_of_color,1);
for i=1:num_of_color
    idx=(C==i);
    para_scanorder{i}=idx;
end
