function fnew = find_targetvalues(obj_space, flow, fup, lambda_objspace, gamma_objspace, Data, old_tv)
%find_targetvalues.m finds the point that maximizes the minimum distance to all points in obj_space
%find the point in the objective space that maximizes the minimum distance between points on Pareto front
%that point will contain target values for each constraint
%have a restriction on that the new point must lie on RBF of Pareto front
%--------------------------------------------------------------------------
%Author information
%Juliane Mueller
%juliane.mueller2901@gmail.com
%--------------------------------------------------------------------------
%input:
%obj_space - non-dominated tionvalue set in objective space
%flow - lower bound for target value search
%fup - upper bound for target value search
%lambda_objspace, gamma_objspace - RBF parameters for approximating the PAreto front
%Data - structure with all up-to-date information
%old_tv - old target values from which we want to stay away
%--------------------------------------------------------------------------
%output:
%fnew - target values for all objectives
%--------------------------------------------------------------------------

warning off
A=[]; %no linear inequalities
b=[];
Aeq = []; %no linear equalities
beq=[];
fbest = inf; %initialize best value so far
fun = @(x)distance(x, obj_space, old_tv); %objective function computing distance of x to all non-dominated points in objective space AND old target values
const = @(x) RBF_const(x, obj_space, lambda_objspace, gamma_objspace, Data); %constraint: new point must lie on RBF approximation surface of Pareto front
xbest = NaN*ones(1,length(flow));
options = optimset('Display','off'); %do not display iteration information
for ii = 1:1%could run optimization more than once to escape local minimum, but may get expensive
    x0 = flow+(fup-flow).*rand(1,length(flow)); %randomly generate initital guess for optimization
    [xout, fout, exitflag, ~] = fmincon(fun,x0,A,b,Aeq,beq,flow,fup,const, options); %MATLAB built-in function fmincon
    if exitflag > 0 && fout < fbest %successful optimization
        fbest = fout;%update best function value and point
        xbest = xout;
    end
end %for
if fbest == inf
    fnew = rand(1,length(flow));
else
    fnew = xbest; % target values for all objectives
end
end%function

function y = distance(x, obj_space, old_tv)  %compute distance between x and [obj_space; old_tv]
allpts=[obj_space;old_tv];
[~,d] = knnsearch(x,allpts);
y = -min(d); %maximize minimum distance = minimize (-minimum distance)
end %function distance

function [c, ceq] = RBF_const(x, obj_space, lambda_objspace, gamma_objspace, Data)
%constraint evaluation: new points should lie on RBF surface of Pareto front
p_space=x(1:Data.nr_obj-1);
[mX,~]=size(p_space); %dimensions of the points where function value should be predicted
points = obj_space(:,1:Data.nr_obj-1);
R = pdist2(p_space,points); %compute pairwise dstances between points in p_space and points. pdist2 is MATLAB built-in function
Phi=R; %linear RBF
p1 = Phi*lambda_objspace; %first part of response surface - weighted sum of distances
p2 = [p_space,ones(mX,1)]*gamma_objspace; % polynomial tail of response surface
yhat=p1+p2; %predicted RBF value of Pareto front
ceq = x(Data.nr_obj)-yhat;
c=[]; %no inequality constraint
end %function constraint
