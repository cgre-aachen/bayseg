function Element=FixElement(Element,MC_knowninfo)
fixed(length(MC_knowninfo),1) = 0;
parfor i = 1:length(MC_knowninfo)
    if MC_knowninfo(i) == 0
        fixed(i) = 0;
    else
        fixed(i) = 1;
    end
end
Element.Fixed = logical(fixed);
end