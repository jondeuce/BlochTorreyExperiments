function [Y_nondom, S_nondom] = check_strict_dominance2(new_vals,new_points, old_vals, old_points)
%%check_strict_dominance2.m checks two sets of points (and their function 
%values) for strict dominance discards dominated points from both sets if necessary
%--------------------------------------------------------------------------
%Author information
%Juliane Mueller
%juliane.mueller2901@gmail.com
%--------------------------------------------------------------------------
%inputs:
%new_vals - matrix with objective function values corresponding to the new points 
%new_points - set of new evaluation points
%old_vals - matrix with objective function values of currently non-dominated solutions
%old_points - set of currently non-dominated points
%--------------------------------------------------------------------------
%outputs:
%Y_nondom - new set of non-dominated function values (may consist of function values of points from new_points and old_points)
%S_nondom - new set of non-dominated points (may consist of points from new_points and old_points)
%--------------------------------------------------------------------------

%now compare remaining points wrt dominance in Y_nondom
old_dom = []; %initialize array of indices of dominated points in old set
new_dom = []; %initialize array of indices of dominated points in new set
for ii =1:size(new_vals,1)
    f1 = new_vals(ii,:);
    for jj =1:size(old_vals,1)
        f2 = old_vals(jj,:);       
        a = find(abs(f1-f2)<1e-6);
        if ~isempty(a)
            val = round(f1(a)*1e6)/1e6;
            f1(a)= val;
            f2(a)=val;
        end      
        if all(f1 <= f2) && any(f1<f2) %new point dominates old one
            old_dom =[old_dom,jj];
        elseif all(f1 == f2)
            new_dom = [new_dom,ii];
            break
        elseif all(f2 <= f1) && any(f2 < f1) %new point is dominated
            new_dom = [new_dom,ii];
            break
        end
    end
end
new_dom = unique(new_dom); %pick only unique indices of points in new set that are dominated
old_dom = unique(old_dom); %pick only unique indices of points in old set that are dominated

%discard all points and function values from old and new set that are dominated
Y_nondom_old = old_vals;
S_nondom_old = old_points;
Y_nondom_old(old_dom,:)=[];
S_nondom_old(old_dom,:)=[];

Y_nondom_new = new_vals;
S_nondom_new = new_points;
Y_nondom_new(new_dom,:)=[];
S_nondom_new(new_dom,:)=[];

%set up matrices with non-dominated points and function values
Y_nondom = [Y_nondom_old; Y_nondom_new];
S_nondom = [S_nondom_old; S_nondom_new];

end %function
