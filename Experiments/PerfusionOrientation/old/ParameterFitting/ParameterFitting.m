function FitResult = ParameterFitting( SimSettings, initguess, lowerbound, upperbound )
%PARAMETERFITTING Executes parameter fitting for simulation.

minimization_time_tic = tic;
display_text( ['Starting Minimization: ' datestr(now)], [], '%', true, [1,1] );

switch upper(SimSettings.MinimizationType)
    case 'FMINCON'
        FitResult	=   FminconFit( SimSettings, initguess, lowerbound, upperbound );
    case 'LSQCURVEFIT'
        FitResult	=   Lsqcurvefit( SimSettings, initguess, lowerbound, upperbound );
    case 'SIMULANNEALBND'
        FitResult	=   SimulatedAnnealingFit( SimSettings, initguess, lowerbound, upperbound );
    otherwise
        error('Unsupported fit type; must be ''fmincon'', ''lsqcurvefit'', or ''simulannealbnd''.');
end

display_text( ['Minimization Finished: ' datestr(now)], [], '%', true, [1,1] );
minimization_time_toc = toc(minimization_time_tic);
display_toc_time( minimization_time_toc, 'Entire Minimization' );    

end

