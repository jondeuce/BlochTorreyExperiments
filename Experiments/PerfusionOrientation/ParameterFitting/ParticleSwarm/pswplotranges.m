function stop = pswplotranges(optimValues,state)
% stop = pswplotranges(optimValues,state)

persistent DateString fig

stop = false; % This function does not stop the solver
switch state
    case 'init'
        DateString = datestr(now,30);
        fig = figure;
        nplot = size(optimValues.swarm,2); % Number of dimensions
        for i = 1:nplot % Set up axes for plot
            subplot(nplot,1,i);
            tag = sprintf('%s__pswplotrange_var_%g',DateString,i); % Set a tag for the subplot
            plot(optimValues.iteration,0,'-k','Tag',tag); % Log-scaled plot
            ylabel(sprintf('Particle %d',i));
        end
        xlabel('Iteration','interp','none'); % Iteration number at the bottom
        subplot(nplot,1,1) % Title at the top
        title('Range of particles by component')
    case 'iter'
        figure(fig);
        nplot = size(optimValues.swarm,2); % Number of dimensions
        for i = 1:nplot
            subplot(nplot,1,i);
            % Calculate the range of the particles at dimension i
            irange = max(optimValues.swarm(:,i)) - min(optimValues.swarm(:,i));
            tag = sprintf('%s__pswplotrange_var_%g',DateString,i);
            plotHandle = findobj(get(gca,'Children'),'Tag',tag); % Get the subplot
            xdata = plotHandle.XData; % Get the X data from the plot
            newX = [xdata optimValues.iteration]; % Add the new iteration
            plotHandle.XData = newX; % Put the X data into the plot
            ydata = plotHandle.YData; % Get the Y data from the plot
            newY = [ydata irange]; % Add the new value
            plotHandle.YData = newY; % Put the Y data into the plot
            set(gca,'ylim',[0,max(newY)]); % adjust ylim
            set(gca,'xlim',[0,optimValues.iteration]); % adjust xlim
        end
        drawnow % Show the plot
        savefig(fig,sprintf('%s__pswplotrange',DateString)); % save the figure
    case 'done'
        % No cleanup necessary
end