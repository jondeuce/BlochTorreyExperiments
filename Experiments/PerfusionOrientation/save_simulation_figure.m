function [ ] = save_simulation_figure( filenames, handles, closefigs, SimSettings )
%SAVE_SIMULATION_FIGURE Saves a file of type 'filetype' to the
%savepath indicated in SimSettings
% 
% INPUT ARGUMENTS:
%   filename:       Name of file to be saved (NOT including full path)
%   handles:        Figure handles to be saved
%   closefigs:      Boolean; closes figures after saving if true
%   SimSettings:	Settings for simulation (contains SimSettings.SavePath)

%==========================================================================
% Parse inputs
%==========================================================================

if isa( filenames, 'char' )
    filenames	=   { filenames };
end

if nargin < 2 || isempty( handles )
    prev	=   get(0, 'ShowHiddenHandles');
    set(0, 'ShowHiddenHandles', 'on');
    handles	=   get(0, 'Children');
    set(0, 'ShowHiddenHandles', prev);
    
    if numel(handles) == 0
        warning( 'No figures are open to save! Exiting...' );
        return
    end
end

if nargin < 3 || isempty(closefigs)
    closefigs	=   false;
end

%==========================================================================
% Save figures
%==========================================================================

if numel(filenames) > numel(handles)
    warning( 'Too many filenames specified; using first %d filenames.', ...
        numel(handles) );
    filenames	=   filenames(1:numel(handles));
elseif numel(handles) > numel(filenames)
    warning( 'Too many figures open; saving the last %d figures only', ...
              numel(filenames) );
	handles     =   handles(1:numel(filenames));
end

for ii = 1:numel(filenames)
    
    filename	=	[ SimSettings.SavePath, '/', filenames{ii} ];
    h           =   handles(ii);
    
    if isvalid(h)
        
        try
            
            % Make figure fullscreen with white background
            if ~strcmpi(get(0, 'defaultfigurewindowstyle'), 'docked')
                set(h, 'Position', get(0, 'Screensize'));
            end
            set(h, 'color', 'w');
            
            % Save .fig file
            saveas(h, filename, 'fig' );
            
            % Save snapshots
            print(h, filename, '-depsc','-r1200');
            print(h, filename, '-dpdf','-r1200');
            print(h, filename, '-dpng');
            
            % (EXPORT_FIG has issues with ghostcript somestimes... not worthwhile,
            % can always go back and save higher quality images later
            %export_fig( filename, '-png', '-eps', '-pdf', '-nocrop' );
            
            if closefigs
                close(h);
            end
            
        catch
            
            % Something went wrong
            warning(me.message);
            
        end
        
    end
    
end

end

