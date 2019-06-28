function parse_iterations(filename)
%PARSE_ITERATIONS Write iterations outputs to a file

fid = fopen(filename, 'w');
iter = 1;
Norm_best = Inf;
R2w_best = -Inf;
fprintf(fid, '%s', 'Timestamp       f-count            f(x)       Best f(x)             R2w        Best R2w');
for s = dir('*.mat')'
    try
        Results = load(s.name);
        if isfield(Results, 'Results')
            Results = Results.Results;
            normfun = @(normfun) perforientation_objfun(Results.params, Results.alpha_range, Results.dR2_Data, Results.dR2, Results.args.Weights, normfun);
            f = normfun(Results.args.Normfun);
            R2w = normfun('R2w');
            Norm_best = min(f, Norm_best);
            R2w_best = max(R2w, R2w_best);
            fprintf(fid, '\n%s%8d%16.8f%16.8f%16.8f%16.8f', s.name(1:15), iter, f, Norm_best, R2w, R2w_best);
            iter = iter + 1;
        end
    catch me
        warning(me.message);
    end
end
fclose(fid);

end