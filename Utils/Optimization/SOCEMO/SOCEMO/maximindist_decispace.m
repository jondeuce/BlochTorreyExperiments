function xnew = maximindist_decispace(Data)
%%maximindist_decispace.m finds a point xnew in the hypercube that maximizes the minimum distance to all already 
%%evaluated points 
%--------------------------------------------------------------------------
%Author information
%Juliane Mueller
%juliane.mueller2901@gmail.com
%--------------------------------------------------------------------------
%input:
%Data - structure with all problem information 
%--------------------------------------------------------------------------
%output:
%xnew - new sample point
%--------------------------------------------------------------------------

A=[];%no nonlinear constraints
b=[];
Aeq = []; %no linear constraints
beq=[];
fbest = inf; %current best guess = inf
fun = @(x)distance(x, Data.S); %objective function (see below)
const = []; %no additional constraints
xbest = NaN*ones(1,Data.dim);
options = optimset('Display','off'); %do not display iteration information
for ii = 1:1%could use several trials because of local optima of distance function, but will become expensive
    x0 = rand(1,Data.dim); %select a random starting point for optimization search
    [xout, fout, exitflag, ~] = fmincon(fun,x0,A,b,Aeq,beq,Data.lb,Data.ub,const,options); %use MATLAB's built-in function fmincon
    if exitflag > 0 && fout < fbest
        fbest = fout;
        xbest = xout;
    end
end %for
if fbest == inf %the optimization did not succeed
    xnew = rand(1,Data.dim); %use random point as new sample point if optimization subproblem doesnt have a solution
else
    xnew = xbest; % target values for all objectives
end
end%function

function y = distance(x, obj_space)
%maximize the minimum distance to already evaluated points (minimize (-minimum distance))
[~,d] = knnsearch(x,obj_space);
y = -min(d);
end %function



