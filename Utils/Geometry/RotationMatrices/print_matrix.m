function print_matrix( A )
%PRINT_MATRIX Prints matrix in C format

if isvector(A)
    A = A(:);
    print_vector(A);
else
    print_mat(A);
end

end

function print_vector(A)

fprintf('{ ');
n = length(A);
for i = 1:n
    fprintf(num2str(A(i),4));
    if i<n; fprintf(', '); end
end
fprintf(' };\n');

end

function print_mat(A)

fprintf('{');
[m,n] = size(A);
for i = 1:m
    fprintf('\t{ ');
    for j = 1:n
        fprintf(num2str(A(i,j),4));
        if j<n; fprintf(', '); end
    end
    if i<m; fprintf(' },\n');
    else fprintf(' }');
    end
end
fprintf('\t};\n');

end
