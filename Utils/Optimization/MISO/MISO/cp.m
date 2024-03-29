function Data =  cp(Data)



% coordinate perturbation sampling
% This function is called by miso.m if the user set sampling = 'cp'
% Candidate points are generated by adding random perturbations to randomly
% selected variables of the best point found so far. A scoring scheme is
% used to select the best candidate point. 
%--------------------------------------------------------------------------
%Author information
%Juliane Mueller
%juliane.mueller2901@gmail.com
%--------------------------------------------------------------------------
%
%Input:
%Data - structure array with all problem information 
%
%Output:
%Data - updated structure array with all problem information
%--------------------------------------------------------------------------


%% parameters and initialization for coordinate perturbation
cp.ncand= min(100*Data.dim,5000); % number of candidate points
cp.xrange = Data.xup - Data.xlow; %variable range in every dimension
cp.minxrange = min(cp.xrange); %smallest variable range
cp.sigma_stdev_default = 0.2*cp.minxrange;    %perturbation range for generating candidate points
cp.sigma_stdev = cp.sigma_stdev_default;     % current perturbation range 
cp.sigma_min=0.2*(1/2)^6*cp.minxrange;  %smallest allowed perturbation range
cp.maxshrinkparam = 5;% maximal number of perturbation range reduction 
cp.failtolerance = max(5,Data.dim); %threshold for consecutively failed improvement trials
cp.succtolerance =3; %threshold for consecutively successful improvement trials
cp.iterctr = 0; % number of iterations
cp.shrinkctr = 0; % number of times perturbation range was reduced 
cp.failctr = 0; % number of consecutive unsuccessful iterations
cp.succctr=0; % number of consecutive successful iterations
cp.weightpattern=[0.3,0.5,0.8,0.95]; %weight pattern for scoring candidate points


%% optimization iterations
while Data.m < Data.maxeval  %do until budget of function evaluations exhausted 
    %compute RBF parameters
    if strcmp(Data.surrogate, 'rbf_c') %cubic RBF
        rbf_flag = 'cub';
        [lambda, gamma] = rbf_params(Data, rbf_flag);
    elseif strcmp(Data.surrogate, 'rbf_l') %linear RBF
        rbf_flag = 'lin';
        [lambda, gamma] = rbf_params(Data, rbf_flag);
    elseif strcmp(Data.surrogate, 'rbf_t') %thin plate spline RBF
        rbf_flag = 'tps';
        [lambda, gamma] = rbf_params(Data, rbf_flag);
    else
        error('rbf type not defined')
    end
    
    cp.iterctr = cp.iterctr + 1; %increment iteration counter
    fprintf('\n Iteration: %d \n',cp.iterctr); %print some user information
    fprintf('\n Number of function evaluations: %d \n', Data.m);
    fprintf('\n Best function value so far: %d \n', Data.fbest);
    
    %select weight for score computation
    mw=mod(cp.iterctr,length(cp.weightpattern));
    if mw==0
        w_r=cp.weightpattern(end);
    else
        w_r=cp.weightpattern(mw);
    end

    % Compute perturbation probability
    pert_p = min(20/Data.dim,1)*(1-(log(Data.m-2*(Data.dim+1)+1)/log(Data.maxeval-2*(Data.dim+1))));
    %create candidate points by perturbing best point found
    CandPoint = repmat(Data.xbest,cp.ncand,1); 
    for ii =1:cp.ncand
        r=rand(1,Data.dim);
        ar = r<pert_p; %indices of variables to be perturbed
        if ~(any(ar)) %if no variable is to be perturbed, randomly select one
            r = randperm(Data.dim);
            ar(r(1))=1;
        end
        for jj =1:Data.dim %go through all dimensions and perturb variable values as necessary
            if ar(jj)==1
                if ismember(jj,Data.integer) %integer perturbation has to be at least 1 unit 
                    rr=randn(1);
                    s_std=sign(rr)*max(1,abs(cp.sigma_stdev*rr));
                else
                    s_std= cp.sigma_stdev*randn(1);
                end
                CandPoint(ii,jj) = CandPoint(ii,jj) +s_std;
                if CandPoint(ii,jj) < Data.xlow(jj)
                    CandPoint(ii,jj) = Data.xlow(jj)+ (Data.xlow(jj)-CandPoint(ii,jj));
                    if CandPoint(ii,jj) >Data.xup(jj)
                        CandPoint(ii,jj) = Data.xlow(jj);
                    end
                elseif CandPoint(ii,jj) > Data.xup(jj)
                    CandPoint(ii,jj) = Data.xup(jj)- (CandPoint(ii,jj)-Data.xup(jj));
                    if CandPoint(ii,jj) <Data.xlow(jj)
                        CandPoint(ii,jj) = Data.xup(jj);
                    end
                end
            end
        end
    end
    %create second group of candidates by uniformly selecting sample points
    CandPoint2=repmat(Data.xlow, cp.ncand,1) + repmat(Data.xup-Data.xlow,cp.ncand,1).*rand(cp.ncand,Data.dim); 
    CandPoint=[CandPoint;CandPoint2];
    CandPoint(:,Data.integer)=round(CandPoint(:,Data.integer)); %round integer variable values
    
    xnew = compute_scores(Data,CandPoint,w_r, lambda, gamma, rbf_flag); %select the best candidate
    clear CandPoint;
    fevalt = tic; %start timer for function evaluation
    fnew = feval(Data.objfunction,xnew); %new function value
    timer = toc(fevalt); %stop timer for function evaluation
    Data.m=Data.m+1; %update the number of function evaluations
    Data.S(Data.m,:)=xnew; %update sample site matrix with new point
    Data.Y(Data.m,1)=fnew; %update vector with function values
    Data.T(Data.m,1) = timer; %update vector with evaluation times

    if fnew < Data.fbest %update best point found so far if necessary
        if (Data.fbest - fnew) > (1e-3)*abs(Data.fbest)
            % "significant" improvement
            cp.failctr = 0; %update fai and success counters
            cp.succctr=cp.succctr+1; 
            improvement=true;
        else
            %no "significant" improvement
            cp.failctr = cp.failctr + 1; %update fai and success counters
            cp.succctr=0;
            improvement=false;
        end  
        xbest_old=Data.xbest;
        Data.xbest = xnew; %best point found so far
        Data.fbest = fnew; %best objective function value found so far
    else
        cp.failctr = cp.failctr + 1; %update fai and success counters
        cp.succctr=0;
        improvement=false;
    end
    
    
    if improvement %if improvement found, do local candidate point search on continuous variables only if integer variables had changed
        if ~all(Data.xbest(Data.integer) - xbest_old(Data.integer)==0) %integer variables changed
            if strcmp(Data.surrogate, 'rbf_c') %cubic RBF
                rbf_flag = 'cub';
                [lambda, gamma] = rbf_params(Data, rbf_flag);
            elseif strcmp(Data.surrogate, 'rbf_l') %linear RBF
                rbf_flag = 'lin';
                [lambda, gamma] = rbf_params(Data, rbf_flag);
            elseif strcmp(Data.surrogate, 'rbf_t') %thin plate spline RBF
                rbf_flag = 'tps';
                [lambda, gamma] = rbf_params(Data, rbf_flag);
            else
                error('rbf type not defined')
            end      
           
            cont_search_imp=false;
            kl=1;
            while kl <= length(cp.weightpattern)
                w_r=cp.weightpattern(kl); %select weight for score computation
                % Perturbation probability
                pert_prob = min(20/Data.dim,1)*(1-(log(Data.m-2*(Data.dim+1)+1)/log(Data.maxeval-2*(Data.dim+1))));
                %create candidate points
                CandPoint = repmat(Data.xbest,cp.ncand,1); 
                for ii =1:cp.ncand
                    r=rand(1,length(Data.continuous));
                    ar=r<pert_prob;
                    if ~(any(ar))
                        r = randperm(length(Data.continuous));
                        ar(r(1))=1;
                    end
                    for jj =1:length(Data.continuous)
                        if ar(jj)==1
                            sig = cp.sigma_stdev*randn(1);
                            CandPoint(ii,Data.continuous(jj)) = CandPoint(ii,Data.continuous(jj)) +sig;
                            if CandPoint(ii,Data.continuous(jj)) < Data.xlow(Data.continuous(jj))
                                CandPoint(ii,Data.continuous(jj)) = Data.xlow(Data.continuous(jj))+ (Data.xlow(Data.continuous(jj))-CandPoint(ii,Data.continuous(jj)));
                                if CandPoint(ii,Data.continuous(jj)) >Data.xup(Data.continuous(jj))
                                    CandPoint(ii,Data.continuous(jj)) = Data.xlow(Data.continuous(jj));
                                end
                            elseif CandPoint(ii,Data.continuous(jj)) > Data.xup(Data.continuous(jj))
                                CandPoint(ii,Data.continuous(jj)) = Data.xup(Data.continuous(jj))- (CandPoint(ii,Data.continuous(jj))-Data.xup(Data.continuous(jj)));
                                if CandPoint(ii,Data.continuous(jj)) <Data.xlow(Data.continuous(jj))
                                    CandPoint(ii,Data.continuous(jj)) = Data.xup(Data.continuous(jj));
                                end
                            end
                        end
                    end
                end
                xnew = compute_scores(Data,CandPoint, w_r, lambda, gamma, rbf_flag); %select the best candidate
                fevalt = tic; % start timer for function evaluation 
                Fselected = feval(Data.objfunction,xnew); %new function value
                timer = toc(fevalt); %stop timer for function evaluation
                Data.m=Data.m+1; %update the number of function evaluations
                Data.T(Data.m,1) = timer; %update the vector with evaluation times
                Data.S(Data.m,:)=xnew; %update the sample site matrix
                Data.Y(Data.m,1)=Fselected; %update the vector with function values
                if (Data.fbest - Fselected) > (1e-3)*abs(Data.fbest) %update best solution found if necessary
                    cp.failctr = 0;
                    cont_search_imp=true;
                    Data.xbest = xnew; %best point found so far
                    Data.fbest = Fselected; %best value found so far
                end
                if kl < length(cp.weightpattern) %update RBF parameters
                    if strcmp(Data.surrogate, 'rbf_c') %cubic RBF
                        rbf_flag = 'cub';
                        [lambda, gamma] = rbf_params(Data, rbf_flag);
                    elseif strcmp(Data.surrogate, 'rbf_l') %linear RBF
                        rbf_flag = 'lin';
                        [lambda, gamma] = rbf_params(Data, rbf_flag);
                    elseif strcmp(Data.surrogate, 'rbf_t') %thin plate spline RBF
                        rbf_flag = 'tps';
                        [lambda, gamma] = rbf_params(Data, rbf_flag);
                    else
                        error('rbf type not defined')
                    end  
                    kl=kl+1;
                elseif kl==length(cp.weightpattern) && cont_search_imp
                    if strcmp(Data.surrogate, 'rbf_c') %cubic RBF
                        rbf_flag = 'cub';
                        [lambda, gamma] = rbf_params(Data, rbf_flag);
                    elseif strcmp(Data.surrogate, 'rbf_l') %linear RBF
                        rbf_flag = 'lin';
                        [lambda, gamma] = rbf_params(Data, rbf_flag);
                    elseif strcmp(Data.surrogate, 'rbf_t') %thin plate spline RBF
                        rbf_flag = 'tps';
                        [lambda, gamma] = rbf_params(Data, rbf_flag);
                    else
                        error('rbf type not defined')
                    end  
                    
                    kl=1;
                    cont_search_imp=false;
                else
                    kl=kl+1;
                end
            end
        end
    end
    
    %check if perturbation range needs to be decreased
    cp.shrinkflag = 1;      
    if cp.failctr >= cp.failtolerance %check how many consecutive failed improvement trials
        if cp.failctr >= cp.failtolerance && cp.shrinkctr >= cp.maxshrinkparam
            cp.shrinkflag = 0;
            disp('Stopped reducing perturbation range because the maximum number of reduction has been reached.');
        end
        cp.failctr = 0;

        if cp.shrinkflag == 1 % decrease perturbation range
            cp.shrinkctr = cp.shrinkctr + 1;
            cp.sigma_stdev =  max(cp.sigma_min,cp.sigma_stdev/2);
            disp('Reducing the perturbation range to half!');
        end
    end
    if cp.succctr>=cp.succtolerance %check if number of consecutive improvements is large enough
        cp.sigma_stdev=2*cp.sigma_stdev;%increase search radius
        cp.succctr=0;
    end
    
end

end%function