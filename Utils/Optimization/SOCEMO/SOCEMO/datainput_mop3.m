function Data= datainput_mop3
%%datainput_mop3.m is a multi-objective test problems example, source:
%Poloni 'Hybrid GA for multiobjective aerodynamic shape optimization', 1997
%optimal solution: x in [-1/sqrt(3), 1/sqrt(3)]
%disconnected Pareto front
%in order to define your own inputs file, copy-paste the structure of this
%file and adjust values as needed
%--------------------------------------------------------------------------
%Author information
%Juliane Mueller
%juliane.mueller2901@gmail.com
%--------------------------------------------------------------------------

Data.dim = 2; %problem dimension must be 2
Data.xlow=-pi*ones(1,Data.dim); %variable lower bounds
Data.xup=pi*ones(1,Data.dim);  %variable upper bounds
Data.nr_obj = 2; %number of objective functions
Data.objfunction=@(x)my_mofun_p3(x);
end %function

function y = my_mofun_p3(x)
global sampledata;
y = zeros(1,2); %initialize objective function value vector

if size(x,2) ~= 2
    error('The number of variables for this function should be exactly 2.');
end

A1 = 0.5*sin(1) - 2*cos(1) + sin(2) - 1.5*cos(2);
A2 = 1.5*sin(1) - cos(1) + 2*sin(2) - 0.5*cos(2);
B1 = 0.5*sin(x(:,1)) - 2*cos(x(:,1)) + sin(x(:,2)) - 1.5*cos(x(:,2));
B2 = 1.5*sin(x(:,1)) - cos(x(:,1)) + 2*sin(x(:,2)) - 0.5*cos(x(:,2));
y(1) = -(-1 - (A1 - B1).^2 - (A2-B2).^2);% min -f(x)
y(2) = -(-1*(x(:,1)+3).^2 - (x(:,2)+1).^2); %min -f(x)

sampledata = [sampledata; x(:)',y(1), y(2)];
end