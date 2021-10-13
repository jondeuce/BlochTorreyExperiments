function [x] = fmg_lap_per(x, h)
%FMG_LAP_PER Discrete Laplacian - central difference approximation with
%periodic boundary conditions
    
    if nargin < 2
        h = 1;
    end
    
    % *********************************************************************
    
    if isa(x, 'double')
        
        if isreal(x)
            x = fmg_lap_per_d(x, h);
        else
            x = fmg_lap_per_cd(x, h);
        end
        
    elseif isa(x, 'single')
        
        if isreal(x)
            x = fmg_lap_per_s(x, h);
        else
            warning('Complex single not working; casting to double and back.');
            x = single(fmg_lap_per_cd(double(x), h));
            %x = fmg_lap_per_sd(x, h);
        end
        
    else
        error('x must be a complex or real single or double array.');
    end
    
end
