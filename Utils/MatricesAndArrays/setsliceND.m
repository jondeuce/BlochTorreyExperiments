function [ A ] = setsliceND( A, idx, dim, B )
%SETSLICEND Sets the slice A( :, :, ..., :, idx, :, ..., : ) of the
% N-dimensional matrix A, where idx is taken at dimension dim, to the 
% array B.

Asiz	=   size(A);

switch numel(Asiz)
    
    case 2
        
        if      dim == 1
            A(idx,:)   =   B;
        else
            A(:,idx)   =   B;
        end
        
    case 3
        
        if      dim == 1
            A(idx,:,:)   =   B;
        elseif	dim == 2
            A(:,idx,:)   =   B;
        else
            A(:,:,idx)   =   B;
        end
        
    case 4
        
        if      dim == 1
            A(idx,:,:,:)   =   B;
        elseif	dim == 2
            A(:,idx,:,:)   =   B;
        elseif	dim == 3
            A(:,:,idx,:)   =   B;
        else
            A(:,:,:,idx)   =   B;
        end
        
    case 5
        
        if      dim == 1
            A(idx,:,:,:,:)   =   B;
        elseif	dim == 2
            A(:,idx,:,:,:)   =   B;
        elseif	dim == 3
            A(:,:,idx,:,:)   =   B;
        elseif	dim == 4
            A(:,:,:,idx,:)   =   B;
        else
            A(:,:,:,:,idx)   =   B;
        end
        
    case 6
        
        if      dim == 1
            A(idx,:,:,:,:,:)   =   B;
        elseif	dim == 2
            A(:,idx,:,:,:,:)   =   B;
        elseif	dim == 3
            A(:,:,idx,:,:,:)   =   B;
        elseif	dim == 4
            A(:,:,:,idx,:,:)   =   B;
        elseif	dim == 5
            A(:,:,:,:,idx,:)   =   B;
        else
            A(:,:,:,:,:,idx)   =   B;
        end
        
    otherwise
        
        N       =   numel(Asiz);
        args	=   [ repmat(':,',1,dim-1), 'idx', repmat(',:',1,N-dim) ];
        eval( ['A(', args, ') = B;'] );
        
end


end

