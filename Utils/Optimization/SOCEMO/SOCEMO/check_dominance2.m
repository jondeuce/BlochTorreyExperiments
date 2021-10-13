function [Y_nondom, S_nondom] = check_dominance2(new_vals,new_points, old_vals, old_points)
%check_dominance2.m checks for dominance relationships between 2 sets of points
%--------------------------------------------------------------------------
%Author information
%Juliane Mueller
%juliane.mueller2901@gmail.com
%--------------------------------------------------------------------------
%input:
%new_vals - function values of new point set
%new_points - new point set
%old_vals - function value of old point set 
%old_points -old point set 
%--------------------------------------------------------------------------
%output:
%Y_nondom - the function values of all non-dominated points from old and new point sets
%S_nondom - the non-dominated points from old and new point sets
%--------------------------------------------------------------------------

old_dom = []; %initialize list of indices of dominated points in old set
new_dom = [];  %initialize list of indices of dominated points in new set
for ii =1:size(new_vals,1)
    f1 = new_vals(ii,:);
    for jj =1:size(old_vals,1)
        f2 = old_vals(jj,:);
        if all(f1 <= f2) && any(f1<f2) %new point dominates old one
              old_dom =[old_dom,jj];
        elseif all(f2 <= f1) && any(f2 < f1) %new point is dominated
              new_dom = [new_dom,ii];
              break
        end
    end
end
new_dom = unique(new_dom); %find unique indices of dominated points in new set
old_dom = unique(old_dom); %find unique indices of dominated points in old set

%construct arrays with non-dominated points and values from old data set
Y_nondom_old = old_vals; 
S_nondom_old = old_points;
Y_nondom_old(old_dom,:)=[];
S_nondom_old(old_dom,:)=[];

%construct arrays with non-dominated points and values from new data set
Y_nondom_new = new_vals;
S_nondom_new = new_points;
Y_nondom_new(new_dom,:)=[];
S_nondom_new(new_dom,:)=[];

%set up matrices with non-dominated points and function values from old and new point sets
Y_nondom = [Y_nondom_old; Y_nondom_new];
S_nondom = [S_nondom_old; S_nondom_new];

end %function
