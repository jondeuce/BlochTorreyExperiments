function [ fig, FileName ] = perforientation_plot( dR2, dR2_all, AllGeoms, args )
%[ fig, FileName ] = perforientation_plot( dR2, dR2_all, AllGeoms, args )
% Plotter for perforientation_fun. dR2 is the curve to be plotted in the
% top most layer, and dR2_all the curves to be plotted in the background;
% for example, dR2 may be the average (over dimension 1) of dR2_all.
% 
% NOTE: dR2_all will not be plotted if it has the same size as dR2 (it will
% be assumed they are equal).

try
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
    
    vec = @(x) x(:);
    roundToInterval = @(xl,d) [floor(xl(1)/d)*d, ceil(xl(2)/d)*d];
    alpha_range = args.xdata;
    alpha_int = [min(alpha_range(:)), max(alpha_range(:))];
    xl = roundToInterval(alpha_int, 5);
    
    % ---- Plot dR2_all in background, unless it is identical to dR2 ---- %
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
            xlim(xl);
        end
    end
    
    % ---- Plot dR2 in foreground ---- %
    if ~isempty(dR2)
        h = plot(alpha_range(:), [args.dR2_Data(:), dR2(:)], '-', ...
            'marker', '+', 'linewidth', 4, 'markersize', 10);
    end
    
    % ---- Set properties, if anything has been plotted ---- %
    if size(dR2_all, 1) > 1 || ~isempty(dR2)
        props = {'fontsize',16,'interpreter','latex'};
        title(title_lines, props{:});
        
        dR2str = '\Delta R_2';
        if strcmpi(args.type,'GRE')
            dR2str = [dR2str, '^*'];
        end
        
        xlabel('$\alpha$ [deg]', props{:});
        ylabel(['$',dR2str,'$ [1/s]'], props{:});
        
        leg = legend(gca, {['$',dR2str,'$ Data'],'Simulated'});
        set(leg, 'location', 'best', props{:});
        xlim(xl);
        
        drawnow
    end
    
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
        str = 'Unable to save figure.\nError message: %s';
        warning(me.identifier, str, me.message);
    end
catch me
    str = 'Unable to plot perfusion curves.\nError message: %s';
    warning(me.identifier, str, me.message);
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
