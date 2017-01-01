function L = formatField(x,y,Field)
grid_matrix = (combvec(1:length(x),1:length(y)))';
size_grid = max(grid_matrix);
N = size(grid_matrix,1);
L = zeros(N,3);
parfor i = 1:N
    coord = [size_grid(2)-(grid_matrix(i,2)-1),grid_matrix(i,1)];
    grid = grid_matrix(i,:);
    L(i,:) = [x(grid(1)),y(grid(2)),Field(coord(1),coord(2))];
end