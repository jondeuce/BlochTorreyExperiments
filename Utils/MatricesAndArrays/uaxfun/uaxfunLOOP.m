function B = uaxfunLOOP( A, ftype, Bsiz ) %#codegen
%UAXFUNLOOP

% Declare A as variable-sized array
coder.varsize('A', size(A), [0 0])

% Declare output variable
B = zeros(Bsiz);

switch ndims(A)
    case 4
        for ii = 1:size(A,1)
            for jj = 1:size(A,2)
                A(ii,jj,:,:) = squeeze(A(ii,jj,:,:));
            end
        end
    case 5
        for ii = 1:size(A,1)
            for jj = 1:size(A,2)
                for kk = 1:size(A,3)
                    A(ii,jj,kk,:,:) = squeeze(A(ii,jj,kk,:,:));
                end
            end
        end
end

end

