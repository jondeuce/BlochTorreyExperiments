function Data= datainput_mop2
%%datainput_mop2.m is a multi-objective test problems example, source:
%Fonesca and Fleming 'multiobjective optimization and multiple constraint handling
% with evolutionary algorithms -Part II: Application example', 1998
%optimal solution: x in [-1/sqrt(3), 1/sqrt(3)]
%nonconvex Pareto front
%in order to define your own inputs file, copy-paste the structure of this
%file and adjust values as needed
%--------------------------------------------------------------------------
%Author information
%Juliane Mueller
%juliane.mueller2901@gmail.com
%--------------------------------------------------------------------------

Data.dim = 2; %problem dimension variable for mop2
Data.xlow=-4*ones(1,Data.dim); %variable lower bounds
Data.xup=4*ones(1,Data.dim);  %variable upper bounds
Data.nr_obj = 2; %number of objective functions
Data.objfunction=@(x)my_mofun_p2(x); %handle to objective function evaluation
end %function

function y = my_mofun_p2(x)
%objective function evaluations
global sampledata;
y = zeros(1,2); %sample data is a global variable and collects all points at whcih we evaluate
%throughout the optimization and the function values; columns 1-Data.dim =
%point, remaining columns = objective function values
%returned objective function values must be in row format

dim = length(x);
y(1) = 1 - exp(-1*sum(((x - (1/sqrt(dim))).^2),2));%first objective - here needs handles to black-box evaluation
y(2) = 1 - exp(-1*sum(((x + (1/sqrt(dim))).^2),2));%second objective

sampledata = [sampledata; x(:)',y(1), y(2)];
end