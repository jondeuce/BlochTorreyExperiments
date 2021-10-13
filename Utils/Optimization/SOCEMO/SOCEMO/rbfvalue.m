function p = rbfvalue(d, rbf_flag)
%%rbfvalue.m computes the rbf value of an input d (the distance between 2 
%points) according to the type of RBF that is selected (cubic, linear, 
%thin-plate spline); the RBF type must be consistent
%--------------------------------------------------------------------------
%Author information
%Juliane Mueller
%juliane.mueller2901@gmail.com
%--------------------------------------------------------------------------
%input:
%d - the distance between 2 points
%rbf_flag - the type of RBF
%--------------------------------------------------------------------------
%output: 
%p - rbf value of the distance d
%--------------------------------------------------------------------------

if strcmp(rbf_flag,'cub')%cubic RBF
    p = d.^3;
elseif strcmp(rbf_flag, 'lin') %linear RBF
    p = d;
elseif strcmp(rbf_flag,'tps') %thin-plate spline RBF
    if d > 0
        p = d.^2.*log(d+realmin);
    else
        p = zeros(size(d));
    end
end%if

end %function
