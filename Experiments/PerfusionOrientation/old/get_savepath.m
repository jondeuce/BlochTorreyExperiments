function [savepath] = get_savepath( SimSettings )
%GET_SAVEPATH Returns the path where the simulation data (as specified by 
% 'SimSettings') is to be saved.
%
%Input arguments:
%   SimSettings:	Simulation settings structure as created by
%                   CALLSIMULATIONMAIN
% 
%Output arguments:
%   savepath:       Path where main simulation results will be saved

% Prefix savepath with rootpath
savepath	=   SimSettings.RootPath;

% Next folder is type of scan simulated
savepath	=   [ savepath, '/', upper(SimSettings.ScanType), '_Experiment' ];

% Next folder is the current date
savepath	=   [ savepath, '/', SimSettings.Date ];

% (Different min types handled within simulation now)
%{
% % If a minimization routine is being run, a separate subfolder is created
% % for each iteration
% if ~isnan( SimSettings.MinimizationType )
%     
%     switch upper( SimSettings.MinimizationType )
%         case 'FMINCON'
%             savepath	=   [ savepath, '/', 'fminsearch_iter_' ];
%         case 'SIMULANNEALBND'
%             savepath	=   [ savepath, '/', 'simulannealbnd_iter_' ];
%         otherwise
%             error( 'Unsupported minimization type; check supported options!' );
%     end
%     
% end
%}

if exist( savepath, 'dir' )
    
    warning( '(%s) Savepath already exists!', upper(mfilename) );
    choice = input( 'Do you wish to overwrite this savepath? [Y/N]: ', 's' );
    
    switch upper(choice)
        case 'Y'
            % do nothing; simulation may proceed
        otherwise
            error( 'Save path ''%s'' already exists and will not be overwritten.', ...
                    savepath );
    end
    
end

end

