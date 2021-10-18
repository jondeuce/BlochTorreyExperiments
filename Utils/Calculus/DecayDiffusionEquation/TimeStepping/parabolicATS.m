function [ x, options ] = parabolicATS( x0, T, stepper, varargin )
%PARABOLICATS Adaptive time stepping solution to the given parabolic PDE

%==========================================================================
% Parse Inputs
%==========================================================================
options	=	parabolicATSoptions( x0, T, varargin{:} );

%==========================================================================
% Perform Adaptive Time Stepping
%==========================================================================
switch upper(options.StepScheme)
    case 'RICHARDSON'
        x	=	AdaptiveRichardsonStepping( x0, T, stepper, options );
    otherwise
        warning( 'Unknown StepScheme option ''%s''. Using Richardson.', ...
            options.StepScheme );
        x	=	AdaptiveRichardsonStepping( x0, T, stepper, options );
end

end

function x = AdaptiveRichardsonStepping( x0, T, stepper, options )

% Settings
dt      =   options.InitialStep;
m       =   options.SubSteps;
rtol	=   options.RelTol;
atol	=   options.AbsTol;

% Initialize variables
t       =	0;
ddt     =	dt/m;
x       =	x0;
dt_list	=   [];

while t < T
        
    % Take large time step dt
    x_mdt      	=	stepper(x,dt);
    
    % Take m small times steps m*ddt
    x_dt      	=   x;
    for ii = 1:m
        x_dt	=   stepper(x_dt,ddt);
    end
    
    % Compute next time step
    abserr      =   norm( x_dt - x_mdt )/(m^2-1);
    ddt_next	=   ddt * sqrt(atol/abserr);
    
    % Check for small time steps
    isTooSmall	=   ( ddt_next/ddt < sqrt(options.MinStep/T) );
    isMinStep	=   ( dt <= options.MinStep );
    
    if ~isTooSmall || isMinStep
        if isMinStep && ~( T-t < dt )
            warning( 'Minimum stepsize (%f) reached.', options.MinStep );
        end
        
        % Update global time
        t       =   t + dt;
        dt_list	=   [ dt_list, dt ];
        
        % Update x using 4th order Richardson Extrapolation
        x	=   (m^2/(m^2-1)) .* x_dt - (1/(m^2-1)) .* x_mdt;
    end
    
    % Update time step
    dt	=   m*ddt_next;
    dt	=   max( dt, options.MinStep );
    dt	=   min( dt, options.MaxStep );
    dt	=   min( dt, T-t );
    ddt	=   dt/m;
    
end

end

