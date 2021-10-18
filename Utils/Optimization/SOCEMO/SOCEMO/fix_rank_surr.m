function Data = fix_rank_surr(Data)
%fix_rank_surr.m is used to augment the current set of non-dominated Pareto front points 
%with more points such that we can approximate it with an RBF surface
%--------------------------------------------------------------------------
%Author information
%Juliane Mueller
%juliane.mueller2901@gmail.com
%--------------------------------------------------------------------------
%inputs:
%Data - structure with all up-to-date problem info
%--------------------------------------------------------------------------
%outputs:
%Data - structure with updated (added) info
%--------------------------------------------------------------------------

global ga_points %this matrix collects all points and function values of evaluations done by MOGA
%compute parameters for all RBF's of objective functions
%surrogate approximation of all objectives
%use MATLAB multiobjective GA to solve the multiobjective surrogate problem
rbf_flag = 'cub'; %use cubic RBF
fix_rank = true; 
std_v = 0.001/2; 

while fix_rank
    ga_points = [];
    [lambda, gamma] = rbf_params_mo(Data, rbf_flag); %compute RBF parameters for every objective
    f_multiobj = @(x)eval_obj(x, lambda, gamma, Data, rbf_flag); %function handle to obtaining surrogate function values
    options = gaoptimset('Generations', 50, 'Display', 'off'); %set options for MOGA of surrogate problem
    %run MATLAB's MOGA
    [~, ~, ~, ~] = gamultiobj(f_multiobj, Data.dim, [], [], [],[], Data.lb, Data.ub, options);

    %determine nondominated points from ALL ga iterations
    points = ga_points(:, 1:Data.dim); %evaluated points
    vals = ga_points(:, Data.dim+1:Data.dim+Data.nr_obj); %corresponding function values
    [points_nondom, ~] = check_dominance(points, vals); %check which ones of all MOGA points are non-dominated
    
    x_out = points_nondom;
    dist_points  = check_distance(x_out, Data.S, Data.tol_same); %remove points from x_out that are too close to Data.S
    while isempty(dist_points) %all points from multiobjective optimization are too close to lready sampled points. 
        std_v = std_v*2;
        x_bypert = Data.S_for_tv + std_v*randn(size(Data.S_for_tv,1),Data.dim); %perturb current non-dominated solutions 
        %reflection on boundaries if perturbed point landed outside lower and upper bounds
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
        dist_points  = check_distance(x_bypert, Data.S, Data.tol_same);  %delete points that are too close to Data.S      
    end
    
    n_new = size(dist_points,1); %numbr of new points (either by MOGA or by perturbation)
    if n_new>2*Data.nr_obj %too many new points
        p = randperm(n_new); %randomly select 2*number of objective points (we do the real evaluation here)
        id_keep = p(1:2*Data.nr_obj);
        x_keep = dist_points(id_keep,:);
        x_reserve = dist_points(p(2*Data.nr_obj+1:end),:);
    else
        x_keep = dist_points;
        x_reserve=[];
    end
    
    xeval = repmat(Data.xlow, size(x_keep,1),1) + repmat(Data.xup-Data.xlow, size(x_keep,1), 1) .* x_keep; %rescale points to true range
    %do expensive evaluation at the new points
    f_true = zeros(size(x_keep,1),Data.nr_obj);
    for ii = 1:size(x_keep,1)
        fvals = feval(Data.objfunction, xeval(ii,:)); %row vector of objective function values
        f_true(ii,:) =fvals;
        Data.S = [Data.S; x_keep(ii,:)]; %update Data structure with new evaluated point
        Data.Y = [Data.Y;fvals]; %update Data structure with new function value
        Data.m = Data.m+1; %update the number of expensive evaluations done (1 evaluation = one evaluation of ALL objectives)
    end
    %check points for dominance
    %if more than one randpoint, check if the dominate each other
    if size(x_keep,1) > 1 %delete all points from x_keep that are too close to each other
        [points_nondom, vals_nondom] = check_strict_dominance(x_keep,f_true);
        rem_points = points_nondom;
        rem_vals = vals_nondom;
    elseif size(x_keep,1) == 1 %only one new point
        rem_points = x_keep;
        rem_vals = f_true;
    end%if

    if size(x_keep,1)>0 %check if strict dominance between old set and new set of points
        [Y_nondom, S_nondom] = check_strict_dominance2(rem_vals,rem_points, Data.Y_for_tv, Data.S_for_tv);
        Data.Y_for_tv = Y_nondom;
        Data.S_for_tv = S_nondom;
    end
    %compute new matrix rank in objective space
    rank_Y_nondom = rank([Data.Y_for_tv(:,1:Data.nr_obj-1), ones(size(Data.Y_for_tv,1),1)]); %must be Data.nr_obj
    if rank_Y_nondom == Data.nr_obj %new matrix satisfies rank condition
	    fix_rank = false;
    else %check if there are reserve points left from random perturbation, iteratively evaluate, check dominance and add to non-dom set
        if ~isempty(x_reserve)
           try_new = true;
           ii = 1;
           while try_new && ii <= size(x_reserve,1) 
               xeval = Data.xlow + (Data.xup-Data.xlow) .* x_reserve(ii,:); %scale point to true range
               fvals = feval(Data.objfunction, xeval); %row vector of expensive objective function values
               Data.S = [Data.S; x_reserve(ii,:)]; %update Data structure
               Data.Y = [Data.Y;fvals];
               Data.m = Data.m+1;
               [Y_nondom, S_nondom] = check_strict_dominance2(fvals,xeval, Data.Y_for_tv, Data.S_for_tv);
               Data.Y_for_tv = Y_nondom;
               Data.S_for_tv = S_nondom;    
               %compute new rank in objective space
               rank_Y_nondom = rank([Data.Y_for_tv(:,1:Data.nr_obj-1), ones(size(Data.Y_for_tv,1),1)]); %must be equal Data.nr_obj
               if rank_Y_nondom == Data.nr_obj %sample new points, hope they are non-dominated and add to matrix
                    try_new = false;
                    fix_rank = false;
               else
                   ii = ii + 1;
               end
           end
        else
            error('Cant find sufficiently many non-dominated points to fit RBF surface for Pareto front')
            %never happened in the tests
            %might have to leave without using the sampling straegy that depends on the surrogate for pareto front
        end
    end
    if size(Data.S,1)>=Data.maxeval  
	error('Ran out of function evaluation budget while trying to fix an RBF approximation of the Pareto front')
        break
    end
end%while
end %function

%function handle for computing the surrogate model predicted objective function value
function y = eval_obj(x, lambda, gamma, Data,rbf_flag)
global ga_points
y = zeros(Data.nr_obj,1); %initialize vector that will get surrogate objective function values
R = pdist2(x,Data.S); %compute pairwise dstances between point x and set of all already sampled points in Data.S. 
%pdist2 is MATLAB built-in function

if strcmp(rbf_flag,'cub') %cubic RBF
    Phi=R.^3;
elseif strcmp(rbf_flag,'lin') %linear RBF
    Phi=R;
elseif strcmp(rbf_flag,'tps') %thin-plate spline RBF
    Phi=R.^2.*log(R+realmin);
    Phi(logical(eye(size(R)))) = 0;
end

for ii = 1:Data.nr_obj %iteratively, make prediction for each objective
    p1 = Phi*lambda(:,ii); %first part of response surface - weighted sum of distances
    p2 = [x,1]*gamma(:,ii); % polynomial tail of response surface
    y(ii)=p1+p2; %predicted function value
end

ga_points = [ga_points; x(:)', y(:)']; %collect all points evaluated by MOGA together with their f-values

end
