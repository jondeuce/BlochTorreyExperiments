function [ B ] = sliceND( A, idx, dim )
%SLICEND Returns the slice A( :, :, ..., :, idx, :, ..., : ) of the
% N-dimensional matrix A, where idx is taken at dimension dim.

Asiz    =   size(A);
if dim > numel(Asiz)
    if idx == 1
        B = A;
        return
    else
        error('Index out of bounds!');
    end
end

switch numel(Asiz)
    
    case 2
        
        if dim  ==  1
            B    =   A(idx,:);
        else
            B    =   A(:,idx);
        end
        
    case 3
        
        if dim  ==  1
            B    =   A(idx,:,:);
        elseif dim == 2
            B    =   A(:,idx,:);
        else
            B    =   A(:,:,idx);
        end
        
    case 4
        
        if dim  ==  1
            B    =   A(idx,:,:,:);
        elseif dim == 2
            B    =   A(:,idx,:,:);
        elseif dim == 3
            B    =   A(:,:,idx,:);
        else
            B    =   A(:,:,:,idx);
        end
        
    case 5
        
        if dim == 1
            B    =   A(idx,:,:,:,:);
        elseif dim == 2
            B    =   A(:,idx,:,:,:);
        elseif dim == 3
            B    =   A(:,:,idx,:,:);
        elseif dim == 4
            B    =   A(:,:,:,idx,:);
        else
            B    =   A(:,:,:,:,idx);
        end
        
    case 6
        
        if dim == 1
            B    =   A(idx,:,:,:,:,:);
        elseif dim == 2
            B    =   A(:,idx,:,:,:,:);
        elseif dim == 3
            B    =   A(:,:,idx,:,:,:);
        elseif dim == 4
            B    =   A(:,:,:,idx,:,:);
        elseif dim == 5
            B    =   A(:,:,:,:,idx,:);
        else
            B    =   A(:,:,:,:,:,idx);
        end
        
    otherwise
        
        N       =   numel(Asiz);
        args    =   [ repmat(':,',1,dim-1), 'idx', repmat(',:',1,N-dim) ];
        eval( ['B = A(', args, ');'] );
        
end


end

