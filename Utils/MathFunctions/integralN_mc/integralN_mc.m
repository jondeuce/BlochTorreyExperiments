function [Q,E,R,info] = integralN_mc(func,varargin)
%%  MONTE CARLO INTEGRATION ROUTINE WITH STRATIFIED AND ANTITHETIC SAMPLING
%
%   integrates over N dimensional domain with possibly infinite bounds. The
%   aim of this program is provide the user with a robust monte carlo
%   integration routine that is able to use stratified sampling as well as
%   antithetic sampling. Further, it can adapt the number of function calls
%   between iterations to efficiently use vectorizations. It can also
%   report results for each iteration or just final results (similar to
%   fminsearch), and for 1 or 2 dimensions it plots results in a graph, if
%   desired.
%   
%   syntax:
%
%   [Q,E,R,info]=integralN_mc(f)
%   [Q,E,R,info]=integralN_mc(f,ran)
%   [Q,E,R,info]=integralN_mc(f,opt)
%   [Q,E,R,info]=integralN_mc(f,Name,Value)
%   [Q,E,R,info]=integralN_mc(f,ran,opt)
%   [Q,E,R,info]=integralN_mc(f,ran,Name,Value)
%   [Q,E,R,info]=integralN_mc(f,ran,opt,Name,Value)
%
%   f is a vectorized function a la f=@(X)sin(X(:,1))+cos(X(:,2))
%   
%   Q       is the integral
%   E       is the absolute error
%   R       is the relative error
%   info    gives a structure with performance results.
%
%   The available name-value pairs and fields of the options are explained 
%   below together with the default values.
%   Note that in the last syntax the name-value pair overrides any argument
%   in the option structure.
%
%   example: 
%   
%   f=@(X)X(:,1).*exp(-X(:,2).^2);
%   Q=integralN_mc(f,[0 1;0 inf]);
%   Q=integralN_mc(f,[0 1;0 inf],'k',4);
%   opt.k=5;
%   Q=integral2_adapt(f,[0 1;0 inf],opt,'k',4);
%
%   The last calculation is done with k=4 as the name-value pair
%   overrides the structure.
%
%   author: Alexander.Dentler at gmail.com, October 9th 2015, all errors
%   are yours to deal with. 
%
%   version 3

%%  SET OPTIONS
ran=[0 1];          %   default range for 1d integration, expand to dx2 
%                       matrix for higher dimensional integration
abstol=1e-8;        %   absolute statistical error tolerance
reltol=1e-5;        %   relative statistical error tolerance
timelimit = inf;    %   sets a time limit (in sec) on integration
N=500000;           %   max no of function evaluations
k=5;                %   no of intervals in each dimension to use stratified
%                       sampling. Reduce for higher dimensions, or set to 1
%                       to turn off stratified sampling altogether.
n=100;              %   no of function calls per iteration
dyn_n=0;            %   evaluates whether no of points per function call 
%                       can be larger
antithetic=1;       %   indicator to use antithetic sampling
seeed=15;           %   seed for random number generator
plt=0;              %   plots subintegrals with relative errors if set to 1
dsp=0;              %   shows result if set to 1, and shows iterative steps 
%                       if set to 2
dsp_renewal=30;     %   renews the header of the iterative display every so steps
%%  DYNAMIC READ IN
t0 = clock; % record initial time
if nargin>1
    list=who;
    opt=[];arg_list=[];
    pos=1;flag_thereismore=1;
    %%  check if first argument gives us a new range
    if isnumeric(varargin{pos})
        if all(size(varargin{pos},2)==2)
            ran=varargin{pos};
        else
            error('Argument list is off. First argument is numeric but does not comply with range format.')
        end
        pos=pos+1;
        flag_thereismore=nargin>pos;
    end
    %%  check if argument gives us a structure
    if flag_thereismore
        if isstruct(varargin{pos})
            opt=varargin{pos};
            pos=pos+1;
            flag_thereismore=nargin>pos;
        end
    end
    %%  check for name-variable pairs
    if flag_thereismore
        if ((nargin-pos)/2)==fix((nargin-pos)/2)
            arg_list=varargin(pos:end);
        else
            error('No of arguments is off.')
        end
    end  
    %%  add option structure variables if they are part of the list
    if ~isempty(opt)
        for i1=1:numel(list)
            if isfield(opt, char(list{i1}))
                eval(horzcat(matlab.lang.makeValidName(char(list(i1))),'=opt.',char(list{i1}),';'));
            end
        end
    end
    %%  add name-value pair arguments if they are part of the list
    if ~isempty(arg_list)
        for i1=1:numel(arg_list)/2
            if ismember(arg_list{(i1-1)*2+1},list) && isnumeric(arg_list{(i1-1)*2+2})
                eval(horzcat(arg_list{(i1-1)*2+1},'=',num2str(arg_list{(i1-1)*2+2}),';'));
            end
        end
    end 
end
%%  ERROR CHECK
if ~(antithetic==1 || antithetic==0)
    warning('Antithetic sampling is either on or off. We switch it off then if you cannot decide.')
    antithetic=0;
end
if k<=0 || k~=fix(k)
    error('Stratified sampling parameter k requires to be a positive integer. Set to 1 to switch it off.')
end
if size(ran,2)~=2
    if size(ran,1)==2
        warning('The range of the domain must be dx2. But we believe you just got confused as the first dimension is 2, so we switch it for you.')
        ran=ran';
    else
        error('The range of the domain must be dx2.')
    end
end
if ~(abstol>0 || reltol>0 || isfinite(timelimit) || isfinite(N))
    warning('There might be no binding stopping criterium.')
end
%%  SET PARAMTERS AND SUBDIVIDING GRID TO STRATIFY
d = size(ran,1);        %  dimension
%   make improper integral proper
for i1=1:d    
    if all(~isfinite(ran(i1,1:2))) %&& all(isfinite(ran(2,1:2)))
        func=@(X)func([X(:,1:(i1-1))  tan(X(:,i1)) X(:,(i1+1):end)]).*sec(X(:,i1)).^2;
        ran(i1,:)=[-1 1]*pi/2;
    elseif ~isfinite(ran(i1,1)) && isfinite(ran(i1,2))% [-inf B]       
        func=@(X)func([X(:,1:(i1-1))  ran(i1,2)+log(X(:,i1))  X(:,(i1+1):end)])./X(:,i1);
        ran(i1,:)=[eps 1];
    elseif isfinite(ran(i1,1)) && ~isfinite(ran(i1,2))% [A inf]       
        func=@(X)func([X(:,1:(i1-1))  ran(i1,1)-log(X(:,i1)) X(:,(i1+1):end)])./X(:,i1);
        ran(i1,:)=[eps 1];
    end
end
V = prod(range(ran'));  % bounding volume (after transformation)
V_sub = V*(1/k)^d;      % bounding volume for subintervals
% k=max(1,floor(exp(log(n/(1+antithetic))/d))); % no of intervals in each dimension
kd=k^d; % total no of rectangular spaces
%   stratifying
cand=1:kd;  % list of all candidates
ran_sub_lb=NaN(kd,d); % lower bound of each rectangular space
ran_sub_ub=NaN(kd,d); % upper bound of each rectangular space
ind_cell=cell(d,1);
for i1=1:(kd)
    [ind_cell{1:d}] = ind2sub(ones(1,d)*k,i1);
    for i2=1:d
        ran_sub_lb(i1,i2)=ran(i2,1)+1/k*(ind_cell{i2}-1)*(ran(i2,2)-ran(i2,1));
        ran_sub_ub(i1,i2)=ran(i2,1)+1/k*(ind_cell{i2})*(ran(i2,2)-ran(i2,1));
    end
end
%   seeding random number generator
rng(seeed); %   note: for this to hold you need to set dyn_n=0;
%   display
if dsp>0
    if antithetic
        disp('Antithetic sampling is turned on.')
    else
        disp('Antithetic sampling is turned off.')
    end
    if k==1
        disp('Stratified sampling is turned off.')
    else
        disp(horzcat('Stratified sampling is turned off with ',num2str(k),...
            ' intervals in each dimension and ',num2str(kd),' independent boxes.'))
    end
    %%
    prnt_fun=@(x1,x2,x3,x4,x5,x6,x7,x8,how)fprintf('%8.3g %9.4g %8.2g %8.2g  %9.4g  %9.4g  %8.2g  % 8.2g   % s  \n',x1,x2,x3,x4,x5,x6,x7,x8,how);
    header=sprintf('\n      I#         Q        E        R         F#    min(F#)     n(F#)      time   comment');
    disp(header)
    prnt_fun(NaN,NaN,abstol,reltol,N,NaN,NaN,timelimit,'thresholds')
end
%%  DYNAMIC LOOP
E=NaN;
R=NaN;
e=NaN(kd,1);
r=NaN(kd,1);
w=ones(kd,1);
points_used_locally=zeros(kd,1);
s=zeros(kd,2);
t1=inf;
iter=0;
n_start=1;
while ~(E<abstol) && ~(R<reltol) && etime(clock,t0)<timelimit && sum(points_used_locally)<=N
    iter=iter+1;
    %%  antithetic sampling
    nantithetic=n/(1+antithetic);
    %%  adaptive stratification
    if kd>1
        if any(points_used_locally<30)
            %%  sample size that gives estimates of error for each bin is
            %   still too small to use errors as weighting scheme
            cand_list_antithetic=mod(n_start-1:(nantithetic-2+n_start),kd)'+1;
            n_start=mod(cand_list_antithetic(end),kd)+1;
            how='preliminary sampling';
        else
            %cand_list_antithetic = randsample(cand,nantithetic,true,w+eps)';
            cand_list_antithetic=(sum(bsxfun(@ge,rand(1,nantithetic),cumsum(w+eps)/sum(w+eps)))+1)';
            how='stratified sampling';
        end
    else
        cand_list_antithetic=ones(nantithetic,1);
        how=' ';
    end
    %%  optimal sample size II
    t2=cputime;
    %%  core evaluation
    if antithetic
        XX=rand(nantithetic,d);
        X=[XX;1-XX];
        cand_list=[cand_list_antithetic;cand_list_antithetic];
    else
        X=rand(nantithetic,d);
        cand_list=cand_list_antithetic;
    end
    X=X.*(ran_sub_ub(cand_list,:)-ran_sub_lb(cand_list,:))+ran_sub_lb(cand_list,:);
    f = func(X);
    if kd>1
        for i1=1:kd
            F=f(cand_list==i1);
            s(i1,:) = s(i1,:) + [sum(F) sum(F.*F)];
        end
    else
        s = s + [sum(f) sum(f.*f)];
    end
    %   keep sample size for each subintegral
    points_used_locally=points_used_locally+histc(cand_list,cand);
    points_used_globally=sum(points_used_locally);
    %%  optimal sample size II
    %   this might result in slower progress because some
    %   variables grow in size and then are faster the second time around
    if dyn_n
        t2=(cputime-t2)/n;
        if t2<t1
            n=n*2;
        end
        t1=t2;
    end
    %%  aggregate evaluation
    %   current absolute error for each rectangular sub-space (std of mean)
    e =V_sub*sqrt(s(:,2) - (s(:,1)).^2./points_used_locally+eps)./points_used_locally;
    %   weight for sampling
    w=e.^2;
    %   current relative error for each rectangular sub-space
    r=sqrt(s(:,2) - (s(:,1)).^2./points_used_locally+eps)./s(:,1);
    %   aggregate absolute error
    E=V/points_used_globally*sqrt( sum(s(:,2))/points_used_globally - (sum(s(:,1))/points_used_globally).^2+eps);
    %   full integral
    Q = sum(V_sub./points_used_locally .* s(:,1));
    %   aggregate relative error
    R = E/abs(Q);
    %   display iterative information
    if dsp>1        
        if mod(iter,dsp_renewal)==0
            disp(header);
            prnt_fun(NaN,NaN,abstol,reltol,N,NaN,NaN,timelimit,'thresholds');
        end
        prnt_fun(iter,Q,E,R,points_used_globally,min(points_used_locally),n,etime(clock,t0),how);
    end
end
%%  REPORT RESULT
%   display results
totalTime = etime(clock,t0);
if dsp==1
    prnt_fun(iter,Q,E,R,points_used_globally,min(points_used_locally),n,totalTime,'final result');    
end
if dsp>0    
    prnt_fun(NaN,NaN,E<abstol,R<reltol,points_used_globally>N,NaN,NaN,totalTime>timelimit,'tolerances met');
end
%   plot error analysis
%   note: this only works in 1 or 2 dimensions
if plt
    q = V_sub./points_used_locally .* s(:,1);
    if d==2
        dim1=(ran_sub_lb(:,1)+ran_sub_ub(:,1))/2;
        dim2=(ran_sub_lb(:,2)+ran_sub_ub(:,2))/2;
        figure
        scatter3(dim1,dim2,q,100,r,'filled')
        xlim(ran(1,:));xlabel('First dimension')
        ylim(ran(2,:));ylabel('Second dimension')
        title('Estimates of sub-integrals. Coloring indicates relative error.')
        colorbar;
    elseif d==1
        dim1=(ran_sub_lb(:,1)+ran_sub_ub(:,1))/2;
        figure
        scatter(dim1,q,100,r,'filled')
        title('Estimates of sub-integrals. Coloring indicates relative error.')
        colorbar;
    else
        warning('Plotting only works in 1 or 2 dimensions.')
    end
end
%   build info structure to pass to user
if nargout>2
    info.n = n;
    info.N = N;
    info.E = E;
    info.R = R;
    info.e = e;
    info.r = r;
    info.iter = iter;
    info.no_intervals_per_d=k;
    info.grid=linspace(0,1,k);
    info.no_rectangular_spaces=kd;
    info.points_used_locally = points_used_locally;
    info.points_used_globally = sum(points_used_locally);
    info.pointdiff = N - sum(points_used_locally);
    info.totalTime = totalTime;
    info.abstolMet = E < abstol;
    info.reltolReq = R < reltol;
    info.timeMet = info.totalTime<timelimit;
    info.sometolReq = (E < abstol) || (R < reltol);
end
end
