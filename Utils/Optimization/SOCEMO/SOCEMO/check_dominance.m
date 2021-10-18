function [points_nondom, vals_nondom] = check_dominance(points, vals)
%check_dominance.m filters out the non-dominated points in "points" based 
%on their function values in "vals". We include points that have the exact 
%same function values
%--------------------------------------------------------------------------
%Author information
%Juliane Mueller
%juliane.mueller2901@gmail.com
%--------------------------------------------------------------------------
%input: 
%points - the set of sample points from which we want to find the non-dominated points
%vals - the function values corresponding to the sample points
%--------------------------------------------------------------------------
%output:
%points_nondom = the non-dominated sample points
%vals_nondom = the corresponding non-dominated function values
%--------------------------------------------------------------------------

id_dominated = []; %initialize list of indices of non-dominated points

for ii = 1:size(points,1)-1 % go through all points and do pairwise comparison of function values
    for jj = ii + 1:size(points,1)
        f1 = vals(ii,:); 
        f2 = vals(jj,:);
        if all(f1 <= f2) && any(f1 < f2) %point ii dominates point jj
            id_dominated = [id_dominated, jj];
        elseif all(f2 <= f1) && any(f2 < f1) %point ii is dominated
            id_dominated = [id_dominated, ii];
            break %go to next ii
        end
    end
end
id_dominated = unique(id_dominated); %only take the unique point indices
points_nondom = points; %initialize list of non-dominated points with list of all points 
points_nondom(id_dominated,:) = []; %delete all dominated points from list of non-dominated points
vals_nondom = vals; %initialize list of non-dominated values with list of all values 
vals_nondom(id_dominated,:) = []; %delete all dominated values from list of non-dominated values

end%function