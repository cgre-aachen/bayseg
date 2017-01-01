function m = findMaxColor(Element)
N = length(Element);
C(N) = 0;
for i = 1:N
    C(i) = Element(i).Color;
end
m = max(C);
end