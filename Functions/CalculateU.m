function Element=CalculateU(Element,U)
N = length(Element.Color);
Element.SelfU=ones(N,1)*U;
end
