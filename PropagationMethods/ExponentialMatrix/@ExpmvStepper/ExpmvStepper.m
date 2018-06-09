classdef ExpmvStepper
    %EXPMVSTEPPER Exponential matrix stepping method.
    
    properties (GetAccess = public, SetAccess = immutable)
        opts % Parsed input options
    end
    
    properties (GetAccess = public, SetAccess = private)
        A         % Matrix or object representing the action of a matrix
        TimeStep  % Time step
        GridSize  % Underlying grid size (may be empty)
        VoxelSize % Underlying voxel dimensions (may be empty)
    end
    
    properties (GetAccess = private, SetAccess = private, Hidden = true)
        isprecomputed  % Flag indicating if M was been precomputed
        selectdegargs  % Parsed arguments for select_taylor_degree
        expmvargs      % Parsed arguments for expmv
        adapttaylor    % Allow adapting taylor steps based on previous iterations
    end
    
    methods
        
        % ----- Constructor ----- %
        function V = ExpmvStepper(t, A, gsize, vdims, varargin)
            % V = ExpmvStepper(t, A, gsize, vsize, varargin)
            % REQUIRED POSITIONAL ARGS:
            % 	A       Matrix or object representing the action of a matrix
            %   t       Time step
            % 	gsize   Underlying grid size (may be empty)
            %   vdims   Underlying voxel dimensions (may be empty)
            %
            % OPTIONAL KEYWORD ARGS:
            %   TODO
            
            [V.opts, V.selectdegargs, V.expmvargs] = bt_expmv_parseinputs(varargin{:});
            
            V.A = A;
            V.TimeStep = t;
            V.GridSize = gsize;
            V.VoxelSize = vdims;
            V.isprecomputed = false;
            
        end
        
        function V = precompute(V,b)
            
            V.expmvargs{1}  = select_taylor_degree(V.A,b,V.selectdegargs{:});
            V.isprecomputed = true;
            
        end
        
        function V = clearPrecomputation(V)
            
            V.expmvargs{1}  = [];
            V.isprecomputed = false;
            
        end
        
        function V = updateMatrix(V,newA)
            
            V.A = newA;
            V = clearPrecomputation(V);
            
        end
        
    end
    
    methods (Static = true)
        % ----- Testing ----- %
        test
    end
    
end

