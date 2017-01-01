function [Element,m]=LRSC(Element)
v_o = ordering(Element);
Element.Color(v_o(1)) = 1;
m = 1;
num_of_Element = length(Element.Color);

for idx=2:num_of_Element
    [flag,k]=findk(Element,v_o(idx),m);
    if flag==1
        Element.Color(v_o(idx)) = k;
    else
        Element.Color(v_o(idx)) = m+1;
        m = m+1;
    end    
end

end