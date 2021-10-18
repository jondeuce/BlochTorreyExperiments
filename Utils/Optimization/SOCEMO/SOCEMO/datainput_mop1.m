function Data= datainput_mop1
%%datainput_mop1.m is a multi-objective test problems example, source:
%Schaffer 'multiple objective optimization with vector evaluated genetic
%algorithms' 1987
%optimal solutions: x in [0,2]
%convex pareto front
%in order to define your own inputs file, copy-paste the structure of this
%file and adjust values as needed
%--------------------------------------------------------------------------
%Author information
%Juliane Mueller
%juliane.mueller2901@gmail.com
%--------------------------------------------------------------------------

Data.dim = 1; %problem dimension variable
Data.xlow=-1e2*ones(1,Data.dim); %variable lower bounds
Data.xup=1e2*ones(1,Data.dim);  %variable upper bounds
Data.nr_obj = 2; %number of objective functions
Data.objfunction=@(x)my_mofun_p1(x); %handle to black-box objective function evaluation
end %function

function y = my_mofun_p1(x)
%objective function evaluations
global sampledata; %sample data is a global variable and collects all points at whcih we evaluate
%throughout the optimization and the function values; columns 1-Data.dim =
%point, remaining columns = objective function values
y = zeros(1,2); %initialize objective function value vector (1 row, 2 columns, 2 = number of objective functions in this example)
%returned objective function values must be in row format

y(1) = sum((x.^2),2); %first objective - here needs handles to black-box evaluation
y(2) = sum(((x-2).^2),2); %second objective

sampledata = [sampledata; x(:)',y(1), y(2)];
end