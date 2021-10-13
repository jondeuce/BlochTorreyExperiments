function xnew = pert_sampling(Data,ncand, pert_p, sigma_stdev, lambda, gamma, w_r)
%pert_sampling.m selects new sample points by perturbing the currently non-dominated points
%we compute scores based on the objective function value predicitons at the perturbed points and
%their distance to already evaluated points
%--------------------------------------------------------------------------
%Author information
%Juliane Mueller
%juliane.mueller2901@gmail.com
%--------------------------------------------------------------------------
%input: 
%Data - structure with all problem information
%ncand - the number of perturbation points we want to create around each currently non-dominated point
%pert_p - the perturbation probability; each parameter va;ue of a non-dominated point will be perturbed with probability pert_p
%sigma_stdev - the standard deviation for the gaussian perturbation N(0, sigma_stdev)
%lambda,gamma - the parameters of the RBF models for each objective
%w_r - the weight (importance) we place on the objective function value prediction 
%--------------------------------------------------------------------------
%output:
%xnew - new sample point
%--------------------------------------------------------------------------

xnew=zeros(size(Data.S_nondom,1),Data.dim); %initialize empty matrix with as many new points as currently non-dominated points
for kk = 1: size(Data.S_nondom,1) %compute candidate points
    CandPoint = repmat(Data.S_nondom(kk,:),ncand,1); 
    R=rand(ncand, Data.dim);
    if Data.dim ==1
        AR =ones(ncand,1);
    else
        AR  = R<pert_p; %AR is matrix with 0 = do not perturb, 1= do perturb
    end
    zerosum_id = find(sum(AR,2)==0);
    for ii =1:length(zerosum_id)
        rp = randperm(Data.dim);
        AR(zerosum_id(ii),rp(1))=1; %at least one variable value must be perturbed
    end
    
    pertpoint = CandPoint + AR.*(sigma_stdev*randn(size(AR))); %do the pertrubation
    for ii =1:Data.dim %reproject values outside [lb, ub] back into the feasible interval
        lesslb = find(pertpoint(:,ii)<Data.lb(ii));
        cpless = pertpoint(lesslb,ii);
        cpless = Data.lb(ii) +(Data.lb(ii)-cpless);
        cpless(cpless>Data.ub(ii)) = Data.lb(ii);
        pertpoint(lesslb,ii) = cpless;        
        overub = find(pertpoint(:,ii)>Data.ub(ii));
        cpover = pertpoint(overub,ii);
        cpover = Data.ub(ii) - (cpover -Data.ub(ii));
        cpover(cpover<Data.lb(ii)) = Data.ub(ii);
        pertpoint(overub,ii) = cpover;
    end

    CandPoint =pertpoint;
    
    %compute scores
    xnew(kk,:) = compute_scores_mo(Data,CandPoint, w_r, lambda, gamma, 'cub');
end
end%function
