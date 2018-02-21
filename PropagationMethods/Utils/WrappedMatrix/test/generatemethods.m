% Automatic generation of method definitions
s = methods('double');

fprintf('        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n');
fprintf('        %% Auto-generated: unary operators \n');
fprintf('        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n');

A = 10 * complex(randn(5),randn(5));

for ii = 1:numel(s)
    try
        func = str2func(s{ii});
        
        if nargin(func) == 1
            B = func(A);
            
            if isequal( size(A), size(B) ) && strcmpi( class(A), class(B) )
                % In-place modification of input
                fprintf('        function [ f ] = %s( f ) %%%s\n', s{ii}, upper(s{ii}));
                fprintf('            f.A = %s( f.A );\n', s{ii});
                fprintf('        end\n\n');
            else
                % Returns some other type of value
                fprintf('        function [ out ] = %s( f ) %%%s\n', s{ii}, upper(s{ii}));
                fprintf('            out = %s( f.A );\n', s{ii});
                fprintf('        end\n\n');
            end
            
        end
        
    catch me
        continue
    end
end

fprintf('        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n');
fprintf('        %% Auto-generated: binary operators \n');
fprintf('        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n');

A = 10 * complex(randn(5),randn(5));
B = 10 * complex(randn(5),randn(5));

for ii = 1:numel(s)
    try
        func = str2func(s{ii});
        
        if nargin(func) == 2
            C = func(A,B);
            
            if isequal( size(A), size(C) ) && strcmpi( class(A), class(C) )
                % Results in a matrix
                fprintf('        function [ h ] = %s( f, g ) %%%s\n', s{ii}, upper(s{ii}));
                fprintf('            fIsWrapped = isa(f, ''WrappedMatrix'');\n');
                fprintf('            gIsWrapped = isa(g, ''WrappedMatrix'');\n');
                fprintf('            if fIsWrapped && gIsWrapped\n');
                fprintf('                h = WrappedMatrix(%s(f.A,g.A));\n', s{ii}); % Wrapped * Wrapped -> Wrapped
                fprintf('            elseif  fIsWrapped && ~gIsWrapped\n');
                fprintf('                h = %s(f.A,g);\n', s{ii}); % Wrapped * double = double
                fprintf('            elseif ~fIsWrapped &&  gIsWrapped\n');
                fprintf('                h = WrappedMatrix(%s(f,g.A));\n', s{ii}); % double * Wrapped = Wrapped
                fprintf('            end\n');
                fprintf('        end\n\n');
            else
                % Returns some other type of value
                fprintf('        function [ out ] = %s( f, g ) %%%s\n', s{ii}, upper(s{ii}));
                fprintf('            fIsWrapped = isa(f, ''WrappedMatrix'');\n');
                fprintf('            gIsWrapped = isa(g, ''WrappedMatrix'');\n');
                fprintf('            if fIsWrapped && gIsWrapped\n');
                fprintf('                out = %s(f.A,g.A);\n', s{ii});
                fprintf('            elseif  fIsWrapped && ~gIsWrapped\n');
                fprintf('                out = %s(f.A,g);\n', s{ii});
                fprintf('            elseif ~fIsWrapped &&  gIsWrapped\n');
                fprintf('                out = %s(f,g.A);\n', s{ii});
                fprintf('            end\n');
                fprintf('        end\n\n');
            end
            
        end
        
    catch me
        continue
    end
end
