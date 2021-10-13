function [points_nondom, vals_nondom] = check_strict_dominance(points, vals)
%check_strict_dominance.m check for strictly dominated sample points by 
%examining the function values in "vals"
%--------------------------------------------------------------------------
%Author information
%Juliane Mueller
%juliane.mueller2901@gmail.com
%--------------------------------------------------------------------------
%input:
%points - list of all sample points
%vals - list of coresponding function values
%--------------------------------------------------------------------------
%output:
%points_nondom - list of strictly non-dominated points
%vals_nondom - list of strictly non-dominated values corresponding to points_nondom
%--------------------------------------------------------------------------

id_dominated = []; %initialize empty list of dominated sample ID
diff_tol=1e-6; %tolerance for function values: if function values differ less than 10e-6, we assume the values are equal

for ii = 1:size(points,1)-1 %go through all points and do pairwise function value comparison
    for jj = ii + 1:size(points,1)
        f1 = vals(ii,:);
        f2 = vals(jj,:);
        a=find(abs(f1-f2)<diff_tol); %check how close the function values are
        if ~isempty(a) %if any 2 values are too close to each other, do small transformation 
            val = round(f1(a)*1e6)/1e6;
            f1(a)= val;
            f2(a)=val;
        end
        if all(f1 <= f2) && any(f1 < f2) %point ii dominates point jj
            id_dominated = [id_dominated, jj];
        elseif all(f1 == f2) %when all function values are the same, consider one of the points dominated
            id_dominated = [id_dominated, ii];
            break %go to next ii
        elseif all(f2 <= f1) && any(f2 < f1) %point ii is dominated by jj
            id_dominated = [id_dominated, ii];
            break %go to next ii
        end
    end
end
id_dominated = unique(id_dominated); %only take the unique point indices
points_nondom = points; %set points_nondom to set of all points and...
points_nondom(id_dominated,:) = []; %...remove all dominated points
vals_nondom = vals; %set vals_nondom to set of all function values and...
vals_nondom(id_dominated,:) = []; %...remove all dominated values

end%function