function xnew = rbf_minus_tv(tv,  Data)
%rbf_minus_tv.m looks for the point in the objective space whose predicted 
%objective function values (predicted with RBF) agree with the target 
%values (multi-objective optimization, but objectives not necessarily 
%conflicting)
%--------------------------------------------------------------------------
%Author information
%Juliane Mueller
%juliane.mueller2901@gmail.com
%--------------------------------------------------------------------------
%input: 
%tv - target value vector, one target value for each objective
%Data - structure with all up-to-date problem information
%--------------------------------------------------------------------------
%output: 
%xnew - the point that minimizes the difference between RBF's of objectives and target values: all((s(x)-tv)=0)
%--------------------------------------------------------------------------

global ga_points %keep all points evaluated by MOGA in matrix
%compute parameters for all RBF's of objective functions
%use MATLAB multiobjective GA to solve the surrogate problem
rbf_flag = 'cub'; %cubic RBF -- can also choose another one here if desired
ga_options = gaoptimset('Generations', 100, 'PopulationSize', 100, ...
    'InitialPopulation', Data.S_nondom, 'Display','off'); %set options for genetic algorithm
ga_points = [];%initialize matrix for GA-evaluated points and function values
[lambda, gamma] = rbf_params_mo(Data, rbf_flag);
f_multiobj = @(x)eval_obj(x, lambda, gamma, Data, rbf_flag, tv); %multi-objective objective function
[x_out, f_out, ~, ~] = gamultiobj(f_multiobj, Data.dim, [], [], [],[], Data.lb, Data.ub, ga_options);

f_sum = sum(abs(f_out),2); %compute the sum of the absolute values of the MOGA solutions
[~,b] = min(f_sum); %find the solution with minimum sum of absolute values 
xnew = x_out(b,:); 

end %function


function y = eval_obj(x, lambda, gamma, Data,rbf_flag, tv)
global ga_points
y = zeros(Data.nr_obj,1); %initialize the vector for objective function values
R = pdist2(x,Data.S); %compute pairwise dstances between point x and S. pdist2 is MATLAB built-in function

if strcmp(rbf_flag,'cub') %cubic RBF
    Phi=R.^3;
elseif strcmp(rbf_flag,'lin') %linear RBF
    Phi=R;
elseif strcmp(rbf_flag,'tps') %thin-plate spline RBF
    Phi=R.^2.*log(R+realmin);
    Phi(logical(eye(size(R)))) = 0;
end

%for each objective function, compute RBF predicted value at x and compute distance to target value
for ii = 1:Data.nr_obj
    p1 = Phi*lambda(:,ii); %first part of response surface - weighted sum of distances
    p2 = [x,1]*gamma(:,ii); % polynomial tail of response surface
    y(ii)=abs(p1+p2-tv(ii)); %predicted function value (RBFvalue-targetValue)
end

ga_points = [ga_points; x(:)', y(:)'];

end
