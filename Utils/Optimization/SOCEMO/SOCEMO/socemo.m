function socemo(datafile,maxeval,setrandseed)% surrogate, n_start, init_design, sampling, own_design)
%motv.m is an optimization method for multi-objective problems that have computationally expensive objective functions. 
%saves the results in form of a structure "Data" into result.mat.
%--------------------------------------------------------------------------
%Author information
%Juliane Mueller
%juliane.mueller2901@gmail.com
%--------------------------------------------------------------------------
%inputs:
%datafile - the name of your datafile containing all relevant problem information (see example) 
%maxeval - the maximum number of evaluations you want to allow, e.g. maxeval = 10 means 10 evaluations of each objective
%setrandseed - random number seed
%--------------------------------------------------------------------------

close all  %close all currently open figures
global sampledata %collect all points and function values
sampledata =[]; %initialize matrix 
rng(setrandseed) %set random number generator seed
gaoptions=gaoptimset('PopulationSize',20,'Display','off'); %options for GA when minimizing bumpiness measure and finding surface min

Data=feval(datafile); % load problem data
Data.lb = zeros(1, Data.dim); %set lower bounds; work in [0,1] while sampling, evaluations at real-scale points
Data.ub = ones(1, Data.dim);%set upper bounds
sigma_stdev = 0.02;% perturbation range multiplier
Data.maxeval  = maxeval; %total number of objective function evaluations; every f(x) will be evaluated 500 times
n_start = 2*(Data.dim + 1); %number of points in initial experimental design
old_tv = []; %collect target values already examined (needed for discontinuous Pareto fronts)
ncand = 500; %number of candidates for perturbation sampling
w_r = 0.95; %weight for response surface criterion in sampling by perturbation

TimeStart=tic;%record total computation time

Data.number_startpoints=n_start; 
Data.tol_same = 0.001*min(Data.ub-Data.lb); %tolerance below which 2 points are considered the same

%% generate initial experimental design
InitialPoints = slhd(Data); %use symmetric latin hypercube sampling to get initial design
Data.S = InitialPoints; %keep the sampler at [0,1], evaluate at rescaled points -- works for continuous variables only

%check rank of Data.S: rank([S,1]) must be dimension +1 
fixrank = false;
if rank([Data.S,ones(size(Data.S,1),1)]) < Data.dim + 1
    fixrank = true;
end 
while fixrank && size(Data.S,1) < Data.maxeval%rank is too small to fit RBF, add random points to design to satisfy rank condition
    n_new = Data.dim+1-rank([Data.S,ones(size(Data.S,1),1)]); %number of additional points needed
    randpoint = rand(n_new,Data.dim);  %generate random points
    dist_points  = check_distance(randpoint, Data.S, Data.tol_same); %compute distance of random points to Data.S, discard points if necessary
    randpoint = dist_points;
    n_left = size(randpoint,1); %number of points left after discarding the ones that are too close
    if rank([[Data.S;randpoint], ones(size(Data.S,1)+n_left,1)]) == Data.dim + 1 %recheck rank of Data.S
        Data.S = [Data.S; randpoint];
        fixrank = false;
    end  
end
    
%% do expensive evaluations --sequentially
Data.m=size(Data.S,1); %number of points in Data.S
Data.Y=zeros(Data.m,Data.nr_obj); %initialize matrix for objective function values, 1st column = 1st objective, 2nd column = 2nd objective, etc
for ii = 1: Data.m %go through all points and evaluate
    xeval = Data.xlow + (Data.xup - Data.xlow).*Data.S(ii,:); %rescale to true range
    fvals = feval(Data.objfunction, xeval); %row vector
    Data.Y(ii, :) = fvals; %row vector of function value into matrix of function values
end

%find the currently non-dominated points. This list includes sample points that may
%have the exact same objective function values
[points_nondom, vals_nondom] = check_dominance(Data.S, Data.Y);

Data.S_nondom = points_nondom; %non-dominated points
Data.Y_nondom = vals_nondom; %non-dominated function values

%% optimization loop
while Data.m < Data.maxeval %sample until we run out of function evaluation budget
    
    %% sampling 1: Target value strategy, find point on RBF-approximated Pareto front
    %  that maximizes the minimum distance to current Pareto optimal points. 
    %strict dominance: discard points that have the exactly same objective function values
    [points_nondom, vals_nondom] = check_strict_dominance(Data.S_nondom, Data.Y_nondom);
    Data.S_for_tv = points_nondom; %use strictly non-dominated points for target-value computations
    Data.Y_for_tv = vals_nondom; %use only strictly non-dominated values for target-value computations
    %express the k-th objective as function of the k-1 objectives (k can be any objective, but we choose 
        %the last objective for ease of implementation)
    rank_Y_nondom = rank([Data.Y_for_tv(:,1:Data.nr_obj-1), ones(size(Data.Y_for_tv,1),1)]); %must be =Data.nr_obj 
        %because we fit an approximation function to the current Pareto front

    if rank_Y_nondom <Data.nr_obj %sample new points, hope they are non-dominated and add to matrix
        fixrank = true; %need to find more non-dominated points to fit RBF of Pareto front
    else
        fixrank = false; %have enough non-dominated values to fit Pareto front approximation
    end

    if fixrank %do MOGA on surrogate problem to create more non-dominated points in order to fit RBF to Pareto front
        Data = fix_rank_surr(Data); %solve a computationally cheap multiobjective surrogate problem with MATLAB's multiobj GA
    end
    if size(Data.S,1) >= Data.maxeval %stop if max number of allowed evaluations reached
        break
    end
    %Fit RBF model to f_k(f_1, f_2, ..., f_(k-1)), i.e. use f_1, f_2, ...,
    %f_(k-1) as 'parameters' and f_k as 'response'
    design_matrix = Data.Y_for_tv(:,1:Data.nr_obj-1); %the first k-1 objectives serve as design matrix
    y_rhs = Data.Y_for_tv(:,end); %the k-th objective serves as right hand side
    distances=pdist2(design_matrix,design_matrix); %compute pairwise dstances between points in design_matrix, pdist2 is MATLAB built-in function
    PairwiseDistance=distances; %use linear RBF      
    [a,b]=size(design_matrix); %determine how many points are in the design matrix and what is the dimension
    clear PHI P A RHS lambda_objspace gamma_objspace %clear old allocations of RBF matrices
    PHI(1:a, 1:a)=PairwiseDistance; 
    P = [design_matrix,ones(a,1)];% [S,ones(maxevals,1)];
    A=[PHI,P;P', zeros(b+1,b+1)]; %set up matrix for solving for parameters
    RHS=[y_rhs;zeros(b+1,1)]; %right hand side of linear system
    params=A\RHS; %compute parameters
    lambda_objspace=params(1:a); %parameters for weights in first part of RBF
    gamma_objspace=params(a+1:end); %parameters of polynomial tail
              
    obj_space = Data.Y_for_tv; %matrix [f1(x1), f2(x1), ..., fm(x1); f1(x2), f2(x2), ..., fm(x2); ...]

    %% sampling in objective space
    % the next sample point should be one that maximizes the min distance
    % between pareto points on pareto front
    fxlow = min(obj_space,[],1); %lower bound = lowest function values
    fxup = max(obj_space,[],1); %upper bound = highest function values
    %compute target values for all objectives by maximizing the minimum
    %distance of points on Pareto front
    target_values = find_targetvalues(obj_space,fxlow,fxup, lambda_objspace, gamma_objspace, Data, old_tv);
    old_tv = [old_tv;target_values];  %update set of target values (to prevent from regenerating same points later   
    
    if ~isempty(target_values)
	%find all points in sample space at which the RBF of each objective assumes the target value
        %computationally cheap multi-objective optimization problem, use MATLAB's MOGA
        xnew = rbf_minus_tv(target_values,  Data);          
        
        %check distance of xnew to already evaluated points, discard if too close 
        dist_points  = check_distance(xnew, Data.S, Data.tol_same);
        xnew = dist_points;
        n_left = size(xnew,1);  %number of points left 
        %% do expensive evaluations --sequentially
        xeval = repmat(Data.xlow, n_left,1) + repmat(Data.xup-Data.xlow, n_left,1).*xnew; %rescale to true range
        ynew=zeros(n_left,Data.nr_obj); %initialize matrix for objective function values
        for ii = 1: n_left
              fvals = feval(Data.objfunction, xeval(ii,:)); %row vector of objective function values
              ynew(ii, :) = fvals; 
              Data.S =[Data.S;xnew(ii,:)]; %update matrices with sample points and sample values
              Data.Y = [Data.Y;ynew(ii,:)];
              Data.m = Data.m + 1; %update function evaluation counter
        end
    end
    %% find the non-dominated points in Data.S
    [S_nondom, Y_nondom] = check_dominance(Data.S, Data.Y);
    Data.Y_nondom = Y_nondom; %update the set of non-dominated points and their function values
    Data.S_nondom = S_nondom;
    
    %% sampling 2: sampling by perturbing points on pareto front (non-dominated points)    
    % Perturbation probability
    pert_p = min(20/Data.dim,1)*(1-(log(Data.m-Data.number_startpoints+1)/log(Data.maxeval-2*(Data.dim+1))));
    [lambda, gamma] = rbf_params_mo(Data, 'cub'); %compute RBF parameters for all objectives
    
    %% new sample points by perturbing non-dominated points
    xnew = pert_sampling(Data,ncand, pert_p, sigma_stdev, lambda, gamma, w_r );
    %check distances of xnew to each others, delete points that are too close to each others
    if size(xnew,1)>1
       ii = 1;
       while ii < size(xnew,1)
           jj = ii+1;
           while jj <=size(xnew,1)
               d = sqrt(sum((xnew(ii,:)-xnew(jj,:)).^2,2));
               if d < Data.tol_same
                   xnew(jj,:)=[];
               else
                   jj = jj+1;
               end
           end
           ii =ii+1;
       end   
    end
    
    %xnew contains a candidate sample point associated with each non-dominated point
    %predict objective function values at the points in xnew to predict dominance
    yhat  = zeros(size(xnew,1), Data.nr_obj); %initialize predicted objective function value matrix
    for kk = 1:size(xnew,1)
        yhat(kk,:) = rbf_prediction_mo(xnew(kk,:),Data,lambda, gamma, 'cub'); %row vector
    end 
    if size(xnew,1)>1 %more than one new sample point
        [points_nondom, vals_nondom] = check_dominance(xnew, yhat); %check which xnew are predicted to dominate each others
        rem_points = points_nondom; %keep going only with the points that are predicted to not domiate each others
        rem_vals = vals_nondom;
    else %only one new point
        rem_points = xnew;
        rem_vals = yhat;
    end

    pred_dom = []; %check if new points are predicted to be dominated by old non-dominated points
    for ii =1:size(rem_points,1)
        for jj = 1:size(Data.Y_nondom,1)
            f1 = rem_vals(ii,:);
            f2 = Data.Y_nondom(jj,:);
            if all(f2 <= f1) && any(f2<f1)
                pred_dom = [pred_dom,ii];
                break
            end
        end
    end
          
    if length(pred_dom) < size(rem_points,1) %not all new points are predicted dominated
        rem_points(pred_dom,:) = []; %discard points that are predicted to be dominated
        %do expensive evaluations and check for dominance
        ynew=zeros(size(rem_points,1),Data.nr_obj); %initialize matrix for new function values
        %scale sample points to true range
        xeval = repmat(Data.xlow,size(rem_points,1),1) + repmat(Data.xup-Data.xlow, size(rem_points,1),1).*rem_points;
        for ii = 1: size(rem_points,1)
              fvals = feval(Data.objfunction, xeval(ii,:)); %row vector
              ynew(ii, :) = fvals;
              Data.S =[Data.S;rem_points(ii,:)]; %update matrices with sample points and sample values
              Data.Y = [Data.Y;ynew(ii,:)];
              Data.m = Data.m + 1; %update counter for function evaluations
        end
        %% find the non-dominated points among the new sample points
        xnew = rem_points;
        if size(xnew,1)>1 
            [points_nondom, vals_nondom] = check_dominance(xnew, ynew);
            rem_points = points_nondom;
            rem_vals = vals_nondom;
        else %only one new point
            rem_points = xnew;
            rem_vals = ynew;
        end
        if ~isempty(rem_points)
            %check if new points dominate old ones
            [Y_nondom, S_nondom] = check_dominance2(rem_vals,rem_points, Data.Y_nondom, Data.S_nondom);
            Data.Y_nondom = Y_nondom;
            Data.S_nondom = S_nondom;    
        end
    end
    
    if size(Data.S,1) >= Data.maxeval %stop if budget of function evaluations exhausted
        break
    end
    

    %% sampling 3: sample at minimum point of each objective function's RBF
    [PHI,P,phi0, pdim]=  rbf_matrices(Data,'cub');
    xnew = zeros(Data.nr_obj, Data.dim); %initialize matrix for new sample points
    [a,b]=size(Data.S); %determine how many points are in S and what is the dimension
    A=[PHI,P;P', zeros(b+1,b+1)]; %set up matrix for solving for parameters
    for ii =1:Data.nr_obj
        %compute RBF parameters for objective function ii
        RHS=[Data.Y(:,ii);zeros(b+1,1)]; %right hand side of linear system
        params=A\RHS; %compute parameters
        lambda_parspace=params(1:a);
        gamma_parspace=params(a+1:end); %parameters of polynomial tail
        minimize_RBF=@(x)rbf_prediction(x,Data,lambda_parspace, gamma_parspace, 'cub');
        %use GA to find the minimum of the RBF surface
        [x_rbf, f_rbf]=ga(minimize_RBF,Data.dim,[],[],[],[],Data.lb,Data.ub,[], gaoptions);
        xnew(ii,:) = x_rbf;
    end %for

    %check distance to previously sampled points and between new points, discard points that are too close             
    dist_points  = check_distance(xnew, Data.S, Data.tol_same);
    xnew = dist_points; %points to evaluate
    n_left = size(xnew,1); %number of points left to evaluate    

    %do new funtion evaluations and check for dominance
    if isempty(xnew) %all points are too close to previously sampled ones --might be in local minima of f(x)'s
        %select the sample point that maximizes the minimum distance to all
        %previously evaluated points
        xnew = maximindist_decispace(Data);
        n_left = size(xnew,1);
    end
    %scale new sample points to true range 
    xeval = repmat(Data.xlow, n_left,1) + repmat(Data.xup-Data.xlow, n_left,1).*xnew; 
    %do expensive function evaluations
    if ~isempty(xnew)
        ynew=zeros(n_left,Data.nr_obj); %initiialize matrix for objective function values
        for ii = 1: n_left
            fvals = feval(Data.objfunction, xeval(ii,:)); %row vector
            ynew(ii, :) = fvals;
            Data.S =[Data.S;xnew(ii,:)];%update matrix with sample points
            Data.Y = [Data.Y;ynew(ii,:)]; %update matrix with objective function values
            Data.m = Data.m + 1; %update counter of function evaluations
        end
        if n_left>1 %check if any of the new points dominate each other
            [points_nondom, vals_nondom] = check_dominance(xnew, ynew);
            rem_points = points_nondom;
            rem_vals = vals_nondom;
        else %only one new point
            rem_points = xnew;
            rem_vals = ynew;
        end
        %check if new points dominate any old ones
        if ~isempty(rem_points)
            [Y_nondom, S_nondom] = check_dominance2(rem_vals,rem_points, Data.Y_nondom, Data.S_nondom);
            Data.Y_nondom = Y_nondom; %update function values of non-dominated points
            Data.S_nondom = S_nondom; 
        end
       
       	if size(Data.S,1) >= Data.maxeval %quit if budget of function evaluations exhausted
            break
        end
    end
    
    %% sampling 4: global search (random candidates throughout the whole variable domain)
    CandPoints = rand(ncand, Data.dim); %generate a large set of random points throughout the whole variable domain
    [lambda, gamma] = rbf_params_mo(Data, 'cub');
    %score candpoints
    xnew = compute_scores_mo(Data,CandPoints,w_r, lambda, gamma, 'cub'); %one new sample point
    xeval = Data.xlow + (Data.xup-Data.xlow).*xnew; %rescale new point to true range
    ynew = feval(Data.objfunction, xeval); %row vector
    Data.S =[Data.S;xnew]; %update sample site matrix
    Data.Y = [Data.Y;ynew]; %update objective function value matrix
    Data.m = Data.m + 1; %update counter for function evaluations

    %check if any of the new points dominate old ones
    dominated_new =[];
    dominated_old=[];
    %find the non-dominated points                               
    for ii = 1:size(Data.Y_nondom,1) 
        f1 = Data.Y_nondom(ii,:); %old 
        f2 = ynew; %new
        if all(f1 <= f2) && any(f1 < f2) %point ii dominates point xnew
            dominated_new = [dominated_new, 1];
            break
        elseif all(f2 <= f1) && any(f2 < f1) %point ii is dominated
            dominated_old = [dominated_old, ii];
        end
    end
                            
    dominated_new = unique(dominated_new); %newly sampled points that are dominated, unique indices
    dominated_old = unique(dominated_old); %indices of old points that are dominated by new ones
    points_old_nondom = Data.S_nondom;
    points_old_nondom(dominated_old,:)=[];
    vals_old_nondom = Data.Y_nondom;
    vals_old_nondom(dominated_old,:) =[];
    points_new_nondom = xnew;
    points_new_nondom(dominated_new,:)=[];
    vals_new_nondom=ynew;
    vals_new_nondom(dominated_new,:)=[];
    %update matrix of non-dominated points and objective function values
    Data.S_nondom = [points_old_nondom; points_new_nondom];
    Data.Y_nondom = [vals_old_nondom; vals_new_nondom];
             
    if size(Data.S,1) >= Data.maxeval %quit if budget of function evaluations exhausted
        break
    end
                                                                    
    %%sampling 5: use MATLAB's Multiobjective genetic algorithm (MOGA) to solve the surrogate problem 
    %%(minimize the surrogate models of all objectives simultaneously)
    Data = solve_surr_problem(Data);
    %check for non-dominated points
    [S_nondom, Y_nondom] = check_dominance(Data.S, Data.Y);
    Data.Y_nondom = Y_nondom;%update information on nondominated points and values
    Data.S_nondom = S_nondom;
              
    disp(sprintf('Number of function evaluations: %d', size(Data.S,1)));  %screen output         
end

Data.totaltime = toc(TimeStart); %total time needed for doing the optimization

save('results.mat', 'Data')

end %function motv
    
