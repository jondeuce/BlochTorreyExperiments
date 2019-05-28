classdef KroneckerPropagator
    %GAUSSIANKERNEL Summary of this class goes here
    %   Detailed explanation goes here
    
    properties ( GetAccess = public, SetAccess = immutable )
        timestep % time step for propagator (unitfull)
        gsize    % grid size (unitless)
        gdims    % grid dimensions (unitfull)
        vsize    % size of subvoxels (i.e. gdims./gsize)
    end
    
    properties ( GetAccess = private )
        dim % dimension of propagator
        Ks  % action matrices
        Es  % precomputed matrix exponentials
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % CLASS CONSTRUCTOR:
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods
        
        function G = KroneckerPropagator(As, dt, gsiz, gdim)
            
            % Default to unit-length isotropic subvoxels
            if nargin < 3
                gdim = gsiz;
            end
            
            % Force grid dimensions and size to be 1xn vectors
            gsiz = gsiz(:).';
            gdim = gdim(:).';
            
            if length(gsiz) ~= length(gdim) || ~any(length(gsiz) == [2,3])
                error('gdims and gsize must both be 2- or 3-element vectors');
            end
            
            % Dimension is simplify length of the input arrays
            ndim = numel(As);
            if ndim > 3
                error('Only supports up to 3 dimensions');
            end
            for ii = 1:ndim
                if ~ismatrix(As{ii}) || ~(size(As{ii},1) == size(As{ii},2)) || ~(length(As{ii}) == gdim(ii))
                    error('Input matrices must be square with proper dimension');
                end
            end
            
            % Assign public class properties
            G.timestep = dt;
            G.gsize = gsiz;
            G.gdims = gdim;
            G.vsize = gdim./gsiz;
            G.dim   = ndim;
            
            % Assign private properties
            G.Ks = As;
            G.Es = cell(1,ndim);
            
        end
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % PUBLIC METHODS:
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods ( Access = public, Static = false )
        
        function G = precompute(G)
            
            for ii = 1:G.dim
                G.Es{ii} = expm(G.timestep * G.Ks{ii});
            end
            
        end
        
        function y = conv(G,x)
            
            if ~isa(G,'Geometry.KroneckerPropagator')
                y = conv(x,G);
                return
            end
            
            if ~isPrecomputed(G)
                G = precompute(G);
            end
            
            switch G.dim
                case 1
                    y = G.Es{1} * x;
                case 2
                    y = G.Es{1} * ((G.Es{2} * x.').');
                case 3
                    y = x;
                    y = permute(y, [3,1,2]);
                    y = reshape(y, G.gdims(3), []);
                    y = G.Es{3} * y;
                    y = reshape(y, G.gdims(3), G.gdims(1), G.gdims(2));

                    y = permute(y, [3,1,2]); % (2,3,1) then (2,3,1); equiv. to (3,1,2)
                    y = reshape(y, G.gdims(2), []);
                    y = G.Es{2} * y;
                    y = reshape(y, G.gdims(2), G.gdims(3), G.gdims(1));
                    
                    y = permute(y, [3,1,2]); % (2,3,1) then (1,2,3); equiv. to (2,3,1)
                    y = reshape(y, G.gdims(1), []);
                    y = G.Es{1} * y;
                    y = reshape(y, G.gdims(1), G.gdims(2), G.gdims(3)); % (1,2,3) is a no-op
                    
                otherwise
                    error('Only 2D and 3D implemented');
            end
            
        end
        
        function b = isPrecomputed(G)
            b = true;
            for ii = 1:G.dim
                b = b && ~isempty(G.Es{ii});
            end
        end
                
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % PUBLIC STATIC METHODS:
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    methods (Static = true)
        % Testing script
        all_tests_passed = test()

        % Kronecker sum
        function C = kronsum(A, B)
            if ~(size(A,1) == size(A,2) && size(B,1) == size(B,2))
                error("Matrices A and B must be square");
            end
            C = kron(A, eye(length(B))) + kron(eye(length(A)), B);
        end

        function G = laplacianPropagator(Dcoeff, dt, GridSize, GridDims)
            [~, ~, Lx] = laplacianSparseOp(GridSize(1), {'P'});
            [~, ~, Ly] = laplacianSparseOp(GridSize(2), {'P'});
            [~, ~, Lz] = laplacianSparseOp(GridSize(3), {'P'});

            h = GridDims ./ GridSize;
            Lx = full(Lx ./ h(1)^2);
            Ly = full(Ly ./ h(2)^2);
            Lz = full(Lz ./ h(3)^2);

            G = Geometry.KroneckerPropagator({Lx,Ly,Lz}, dt, GridSize, GridDims);
        end
        
    end
    
end
