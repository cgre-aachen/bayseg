function [gl_Element,gl_field_value,F] = Global_preprocess(F)

addpath('./Functions');
for i = 1:length(F)
    order = 1;
    if ~isfield(F{i},'Element');
        F{i}.Element = constructElements(F{1}.ux1,F{1}.uy1,0,order);
        F{i}.Element = detectNeighborDirection(F{i}.Element,2);
    end
    if ~isfield(F{i},'field_value');
        F{i}.field_value = cat(2,retrieve(F{i}.Element,F{i}.NDVI_z_score),retrieve(F{i}.Element,F{i}.log_weighted_EMI_z_score));
        F{i}.field_value(isnan(sum(F{i}.field_value,2)),:) = NaN;
    end
    if i == 1
        gl_Element.Color = F{i}.Element.Color;
        gl_Element.Neighbors = F{i}.Element.Neighbors;        
        gl_Element.Direction = F{i}.Element.Direction;
        gl_field_value = F{i}.field_value;     
    else
        n_row = length(gl_Element.Color);
        n_extra = length(F{i}.Element.Color);
        gl_Element.Color = cat(1,gl_Element.Color,F{i}.Element.Color);
        temp = F{i}.Element.Neighbors;
        for j = 1:n_extra
            temp{j} = temp{j} + n_row;
        end
        gl_Element.Neighbors = cat(1,gl_Element.Neighbors,temp);        
        gl_Element.Direction = cat(1,gl_Element.Direction,F{i}.Element.Direction);
        gl_field_value = cat(1,gl_field_value,F{i}.field_value);
    end    
end