function simpgif(p,t,expr,bcol,icol,nodes,tris,facecol,filename,caxistype,titles,totaltime,imsize,imscale)

%   Copyright (C) 2004-2012 Per-Olof Persson. See COPYRIGHT.TXT for details.

if nargin == 0
    simpgif_example;
    return
end

if nargin < 14; imscale = 1; end
if nargin < 13; imsize = [1, 1]; end
if nargin < 12; totaltime = 10; end
if nargin < 11; titles = []; end
if nargin < 10; caxistype = 'ALL'; end
if nargin < 9; filename = 'test.gif'; end

if ~isempty(titles)
    if ischar(titles); titles = {titles}; end
    if numel(titles) == 1; titles = repmat(titles,size(facecol,2),1); end
end

dim=size(p,2);
switch dim
    case 2
        if nargin<4 || isempty(bcol), bcol=[.8,.9,1]; end
        if nargin<5 || isempty(icol), icol=[0,0,0]; end
        if nargin<6, nodes=0; end
        if nargin<7, tris=0; end
        if nargin<8, facecol=[]; end
        
        fig = figure( ...
            'visible', 'off', 'windowstyle', 'normal', 'units', 'normalized', ...
            'outerposition', [0, 0, imsize]);
        hold on
        
        if isempty(facecol)
            trimesh(t,p(:,1),p(:,2),0*p(:,1),'facecolor',bcol,'edgecolor','k');
        else
            s = trisurf(t,p(:,1),p(:,2),facecol(:,1),facecol(:,1),'edgecolor','k','facecolor','interp');
            colorbar
        end
        if nodes==1
            line(p(:,1),p(:,2),'linest','none','marker','.','col',icol,'markers',24);
        elseif nodes==2
            for ip=1:size(p,1)
                txtpars={'fontname','times','fontsize',12};
                text(p(ip,1),p(ip,2),num2str(ip),txtpars{:});
            end
        end
        if tris==2
            for it=1:size(t,1)
                pmid=mean(p(t(it,:),:),1);
                txtpars={'fontname','times','fontsize',12,'horizontala','center'};
                text(pmid(1),pmid(2),num2str(it),txtpars{:});
            end
        end
        view(2)
        axis equal
        axis off
        ax=axis;
        axis(ax*1.001);
        
        % Generate gif
        caxisval = get_caxis(caxistype, facecol);
        colorbar(gca(fig), 'fontsize', 12)
        if ~isempty(caxisval)
            caxis(caxisval);
        end
        if ~isempty(titles)
            title(gca(fig), titles{1}, 'fontsize', 12);
        end
        
        gif(filename, 'frame', gca(fig), 'Scale', imscale, 'DelayTime', totaltime / size(facecol,2), 'nodither');
        for ii = 2:size(facecol,2)
            set(s, 'FaceVertexCData', facecol(:,ii));
            if ~isempty(caxisval)
                caxis(caxisval);
            end
            if ~isempty(titles)
                title(gca(fig), titles{ii}, 'fontsize', 12);
            end
            gif;
        end
    otherwise
        error('Unimplemented dimension.');
end

end

function [caxisval] = get_caxis(caxistype, facecol)

if ischar(caxistype)
    switch upper(caxistype)
        case 'FIRST'
            caxisval = [min(vec(facecol(:,1))) max(vec(facecol(:,1)))];
        case 'ALL'
            caxisval = [min(vec(facecol)) max(vec(facecol))];
        otherwise
            caxisval = [];
    end
else
    if isempty(caxistype)
        caxisval = [];
    else
        caxisval = caxistype;
    end
end

end

function simpgif_example

p = randn(50,2);
t = delaunayn(p);

f = @(p,s) exp(-(p(:,1).^2 + p(:,2).^2)/(2*s));
s = linspace(0.1, 5.0, 20);
a = cell(numel(s), 1);
z = zeros(size(p,1), numel(s));
for ii = 1:numel(s)
    z(:,ii) = f(p, s(ii));
    a{ii} = ['sigma = ', sprintf('%.4f', s(ii))];
end

fname = [datestr(now,30), '.test.gif'];
simpgif(p,t,[],[0.8,0.9,1],[0,0,0],0,0,z,fname,[0,1],a);

end