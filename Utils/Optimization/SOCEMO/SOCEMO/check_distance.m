function dist_points  = check_distance(set_new, set2, dist_tol_same)
%check_distance.m computes the distance between two sets of points and 
%deletes points from set_new that are too close to %already evaluated points
%--------------------------------------------------------------------------
%Author information
%Juliane Mueller
%juliane.mueller2901@gmail.com
%--------------------------------------------------------------------------
%input:
%set_new - a new set of points that might want to join our old set of points
%set2 - the old point set, we can't change this anymore
%dist_tol_same - the tolerance for the distance below which we consider two points the same
%--------------------------------------------------------------------------
%output:
%dist_points - the remaining set of point after deleting the ones that are too close to set2
%--------------------------------------------------------------------------

%there is more than one point in set_new, pairwise comparison of points in set_new to themselves, remove "near-replicates"
if size(set_new,1) >1 
    ii = 0;
    while ii < size(set_new,1)
        ii = ii+1;
        jj=ii+1;
        while jj <= size(set_new,1)
            if sqrt(sum((set_new(ii,:) - set_new(jj,:)).^2)) < dist_tol_same
                 set_new(jj,:) =[];
            else
                 jj = jj+1;
            end
        end
    end
end

%compare the remaining points in set_new to old data set "set2"
n_new = size(set_new,1);
del_new=[];
for ii=1:n_new
    [~, knn_dist] = knnsearch(set2, set_new(ii,:));
    if min(knn_dist) < dist_tol_same
        del_new=[del_new,ii];
    end
end

set_new(del_new,:) = []; %delete points fromo set_new that are too close to set2
dist_points = set_new;
end%function
