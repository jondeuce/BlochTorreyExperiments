function pareto_plot(Data)
%pareto_plot.m plots the pareto front for problems with 2 or 3 objective
%functions.
%--------------------------------------------------------------------------
%Author information
%Juliane Mueller
%juliane.mueller2901@gmail.com
%--------------------------------------------------------------------------
%input:
%Data - structure with all problem information after optimization has
%finished (Data from loaded results file, e.g., load results.mat)
%--------------------------------------------------------------------------
close all
if Data.nr_obj == 2 %plot pareto front for 2 objectives
    figure
    set(gca,'Fontsize',20)
    hold on
    plot(Data.Y_nondom(:,1), Data.Y_nondom(:,2),'r.', 'markersize', 6)
    xlabel('Objective 1')
    ylabel('Objective 2')
    legend('Pareto front')
    
elseif Data.nr_obj ==3%plot pareto front for 3 objectives
    figure
    set(gca,'Fontsize',20)
    %hold on
    scatter3(Data.Y_nondom(:,1), Data.Y_nondom(:,2),Data.Y_nondom(:,3),'filled')
    xlabel('Objective 1')
    ylabel('Objective 2')
    zlabel('Objective 3')
    legend('Pareto front')
else %error when we have more than 3 objectives
    error('I can''t plot a Pareto front for more than 3 objectives')
end