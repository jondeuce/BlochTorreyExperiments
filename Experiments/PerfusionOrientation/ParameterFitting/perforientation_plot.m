function [ fig, FileName ] = perforientation_plot( dR2, dR2_all, AllGeoms, args )
%PERFORIENTATION_PLOT Plotter for perforientation_fun

if isempty(args.TitleLinesFun)
    [title_lines] = title_lines_fun( dR2, dR2_all, AllGeoms, args );
else
    [title_lines] = args.TitleLinesFun( dR2, dR2_all, AllGeoms, args );
end

title_str = title_lines{1};
for ii = 2:numel(title_lines)
    title_str = strcat(title_str, ', ', title_lines{ii});
end

FileName = strrep( title_str, ' ', '' );
FileName = strrep( FileName, '^', '' );
FileName = strrep( FileName, '$', '' );
FileName = strrep( FileName, '/', '' );
FileName = strrep( FileName, '\%', '' );
FileName = strrep( FileName, '=', '-' );
FileName = strrep( FileName, ',', '__' );
FileName = strrep( FileName, '--', '-m' ); % double dash is a dash and a minus sign
FileName = strrep( FileName, '.', 'p' );
FileName = [datestr(now,30), '__', FileName];

fig = figure; set(gcf,'color','w'); hold on

alpha_range = args.xdata;
if size(dR2_all, 1) > 1
    % More than one dR2 to plot
    N_dR2 = size(dR2_all, 1);
    cmap = flipud(jet(N_dR2));
%     cmap = jet(N_dR2);
%     cmap = flipud(plasma(N_dR2+5));
%     cmap = cmap(5+1:end, :);
    h_all = plot(alpha_range(:), dR2_all.', '-.');
    for ii = 1:N_dR2
        set(h_all(ii),'color',cmap(ii,:));
    end
end
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

% ---- Save Figure ---- %
try
    if args.PlotFigs && args.SaveFigs
        FigTypes = args.FigTypes;
        [v,ix] = ismember('fig',FigTypes);
        if v
            savefig(fig, FileName);
        end
        FigTypes = FigTypes([1:ix-1,ix+1:end]);
        if ~isempty(FigTypes)
            FigTypes = strcat('-', FigTypes);
            export_fig(FileName, FigTypes{:});
        end
    end
catch me
    warning('Unable to save figure.\nError message: %s', me.message);
end

end

function [title_lines] = title_lines_fun( dR2, dR2_all, AllGeoms, args )

CA = args.params(1);
iBVF = args.params(2);
aBVF = args.params(3);
BVF = iBVF + aBVF;
iRBVF = iBVF/BVF;
alpha_range = args.xdata;

vec = @(x) x(:);
R2  = perforientation_objfun( args.params, alpha_range, args.dR2_Data, dR2(end,:), 'uniform', 'R2' );
R2w = perforientation_objfun( args.params, alpha_range, args.dR2_Data, dR2(end,:), args.Weights, 'R2w' );

Nfun = args.Normfun;
Funval = perforientation_objfun( args.params, alpha_range, args.dR2_Data, dR2(end,:), args.Weights, Nfun );

avg_rminor = mean(vec([AllGeoms.Rminor_mu]));
avg_rmajor = mean(vec([AllGeoms.Rmajor]));

title_lines = {...
    sprintf('iBVF = %.4f%%, aBVF = %.4f%%, CA = %.4f, BVF = %.4f%%, iRBVF = %.2f%%', iBVF*100, aBVF*100, CA, BVF*100, iRBVF*100 ), ...
    sprintf('N = %d, Rmajor = %.2fum, Rminor = %.2fum, R2 = %.4f, R2w = %.4f, %s = %.4f', args.Nmajor, avg_rmajor, avg_rminor, R2, R2w, Nfun, Funval) ...
    };
title_lines = cellfun(@(s)strrep(s,'%','\%'),title_lines,'uniformoutput',false);

end
