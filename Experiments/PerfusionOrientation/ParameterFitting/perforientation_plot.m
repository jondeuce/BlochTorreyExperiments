function [ fig, FileName ] = perforientation_plot( dR2, dR2_all, AllGeoms, args )
%PERFORIENTATION_PLOT Plotter for perforientation_fun

CA = args.params(1);
iBVF = args.params(2);
aBVF = args.params(3);
BVF = iBVF + aBVF;
iRBVF = iBVF/BVF;
alpha_range = args.xdata;

vec = @(x) x(:);
%res = sqrt(sum((vec(dR2_Data)-vec(dR2(end,:))).^2)/numel(xdata));
res = norm( vec( args.dR2_Data - dR2(end,:) ) )^2;
avg_rminor = mean(vec([AllGeoms.Rminor_mu]));
avg_rmajor = mean(vec([AllGeoms.Rmajor]));

title_lines = {...
    sprintf('iBVF = %.4f%%, aBVF = %.4f%%, CA = %.4f, BVF = %.4f%%, iRBVF = %.2f%%', iBVF*100, aBVF*100, CA, BVF*100, iRBVF*100 ), ...
    sprintf('N = %d, Rmajor = %.2fum, Rminor = %.2fum, L2-residual = %.4f', args.Nmajor, avg_rmajor, avg_rminor, res) ...
    };
title_lines = cellfun(@(s)strrep(s,'%','\%'),title_lines,'uniformoutput',false);
title_str = [title_lines{1},', ',title_lines{2}];

FileName = strrep( title_str, ' ', '' );
FileName = strrep( FileName, '\%', '' );
FileName = strrep( FileName, '=', '-' );
FileName = strrep( FileName, ',', '__' );
FileName = strrep( FileName, '.', 'p' );
FileName = [datestr(now,30), '__', FileName];

fig = figure; set(gcf,'color','w'); hold on

plot(alpha_range(:), dR2_all.', '-.');
h = plot(alpha_range(:), [args.dR2_Data(:), dR2.'], '-', ...
    'marker', '+', 'linewidth', 4, 'markersize', 10);

props = {'fontsize',14,'interpreter','latex'};
title(title_lines, props{:});

dR2str = '\Delta R_2';
if strcmpi(args.type,'GRE')
    dR2str = [dR2str, '^*'];
end

xlabel('$\alpha$ [deg]', props{:});
ylabel(['$',dR2str,'$ [1/s]'], props{:});

leg = legend(h, {['$',dR2str,'$ Data'],'Simulated'});
set(leg, 'location', 'best', props{:});

drawnow

end

