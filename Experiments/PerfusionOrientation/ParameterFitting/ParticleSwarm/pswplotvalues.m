function stop = pswplotvalues(optimValues,state)
% stop = pswplotvalues(optimValues,state)

persistent DateString fig

stop = false; % This function does not stop the solver
nplot = size(optimValues.swarm,2); % Number of dimensions
npart = size(optimValues.swarm,1); % Number of particles

switch state
    case 'init'
        DateString = datestr(now,30);
        fig = figure;
        for i = 1:nplot % Set up axes for plot
            subplot(nplot,1,i);
            plot(0,optimValues.swarm(:,i).');
            ylabel(sprintf('Variable %d',i));
        end
        xlabel('Iteration','interp','none'); % Iteration number at the bottom
        subplot(nplot,1,1) % Title at the top
        title('Particles values')
    case 'iter'
        figure(fig);
        for i = 1:nplot
            subplot(nplot,1,i);
            % Calculate the range of the particles at dimension i
            idata = optimValues.swarm(:,i).';
            curAxes = gca;
            plotHandles = get(curAxes, 'children');
            xdata = [plotHandles(1).XData(:); optimValues.iteration];
            ydata = [reshape([plotHandles.YData], optimValues.iteration, npart); idata];
            plot(xdata, ydata);
            %title('Particles values')
            set(curAxes,'ylim',[min(ydata(:)),max(ydata(:))]); % adjust ylim
            set(curAxes,'xlim',[0,optimValues.iteration]); % adjust xlim
            ylabel(sprintf('Variable %d',i));
        end
        subplot(nplot,1,1) % Title at the top
        title('Particles values')
        drawnow % Show the plot
        savefig(gcf,sprintf('%s__pswplotvalue',DateString)); % save the figure
    case 'done'
        % No cleanup necessary
end