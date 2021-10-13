function Data = solve_surr_problem(Data)
%%solve_surr_problem.m solves a multi-objective optimization problem in which we minimize the RBF
%%approximations of the objective functions simultaneously; only selected solutions will be used for 
%%expensive evaluation
%--------------------------------------------------------------------------
%Author information
%Juliane Mueller
%juliane.mueller2901@gmail.com
%--------------------------------------------------------------------------
%input:
%Data - structure with all problem information
%--------------------------------------------------------------------------
%output:
%Data - updated structure with all problem information
%--------------------------------------------------------------------------

global ga_points %collect all points that were evaluated by the MOGA
%compute parameters for all RBF's of objective functions
%use MATLAB multiobjective GA to solve the surrogate problem
rbf_flag = 'cub'; %cubic RBF

%set options for MATLAB's MOGA
ga_options = gaoptimset('Generations', 100, 'PopulationSize', 100, 'InitialPopulation', Data.S_nondom, 'Display','off');
std_v = 0.001/2;%perturbation value; in case MOGA does not generate a solution, we have to find a new point by perturbation strategy

ga_points = [];%initialize matrix

[lambda, gamma] = rbf_params_mo(Data, rbf_flag); %compute parameters of RBF surrogates for all objectives
f_multiobj = @(x)eval_obj(x, lambda, gamma, Data, rbf_flag); %define the multi-objective objective function
%use MATLAB's MOGA to solve the problem
[x_out, ~, ~, ~] = gamultiobj(f_multiobj, Data.dim, [], [], [],[], Data.lb, Data.ub, ga_options);

dist_points  = check_distance(x_out, Data.S, Data.tol_same); %check if any point in x_out is too close to S
except_c = 0;
while isempty(dist_points) %point of multiobjective optimization is too close to lready sampled points. 
    except_c = except_c+1;
    std_v = std_v*2; %standard deviation for perturbation when creating new points from currently non-dominated
    x_bypert = Data.S_nondom + std_v*randn(size(Data.S_nondom,1),Data.dim);
    %reflection on boundaries if perturbation point fell outside the variable bounds
    for ii = 1:size(x_bypert,1)
        for jj = 1:Data.dim
            if x_bypert(ii,jj) < Data.lb(jj)
                x_bypert(ii,jj) = Data.lb(jj)+ (Data.lb(jj)-x_bypert(ii,jj));
                if x_bypert(ii,jj) >Data.ub(jj)
                    x_bypert(ii,jj) = Data.lb(jj);
                end
            elseif x_bypert(ii,jj) > Data.ub(jj)
                x_bypert(ii,jj) = Data.ub(jj)- (x_bypert(ii,jj)-Data.ub(jj));
                if x_bypert(ii,jj) <Data.lb(jj)
                    x_bypert(ii,jj) = Data.ub(jj);
                end
            end
        end
    end
    dist_points  = check_distance(x_bypert, Data.S, Data.tol_same);  %delete points from x_bypert that are too close to S
    if isempty(dist_points) && except_c > 3 %try creating new points by perturbation 3 times, if all successless, use the point
	    %that maximizes the minimum distance to already sampled points
        dist_points = maximindist_decispace(Data);
    end
end
    

n_new = size(dist_points,1); %determine number of new points
if n_new>2*Data.nr_obj %if there are too many new points, randomly select 2*#objectives points from the list
    p = randperm(n_new);
    id_keep = p(1:2*Data.nr_obj);
    x_keep = dist_points(id_keep,:);
else
    x_keep = dist_points;
end
xeval = repmat(Data.xlow, size(x_keep,1),1) +repmat(Data.xup-Data.xlow, size(x_keep,1), 1).*x_keep; %rescale new point to true range
%do expensive evaluation at the new points
f_true = zeros(size(x_keep,1),Data.nr_obj);
for ii = 1:size(x_keep,1)
    fvals = feval(Data.objfunction, xeval(ii,:)); %row vector
    f_true(ii,:) =fvals;
    Data.S = [Data.S; x_keep(ii,:)]; %update data matrices
    Data.Y = [Data.Y;fvals];
    Data.m = Data.m+1;
end

end %function


function y = eval_obj(x, lambda, gamma, Data,rbf_flag)
%use RBF surrogate models to predict function values at x
global ga_points
y = zeros(Data.nr_obj,1);
R = pdist2(x,Data.S); %compute pairwise dstances between point x and S. pdist2 is MATLAB built-in function

if strcmp(rbf_flag,'cub') %cubic RBF
    Phi=R.^3;
elseif strcmp(rbf_flag,'lin') %linear RBF
    Phi=R;
elseif strcmp(rbf_flag,'tps') %thin-plate spline RBF
    Phi=R.^2.*log(R+realmin);
    Phi(logical(eye(size(R)))) = 0;
end


for ii = 1:Data.nr_obj
    p1 = Phi*lambda(:,ii); %first part of response surface - weighted sum of distances
    p2 = [x,1]*gamma(:,ii); % polynomial tail of response surface
    y(ii)=p1+p2; %predicted function value
end

ga_points = [ga_points; x(:)', y(:)']; %collect all points and their predicted function values

end
