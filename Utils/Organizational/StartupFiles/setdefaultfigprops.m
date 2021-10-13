function setdefaultfigprops
%SETDEFAULTFIGPROPS Sets global figure properties for nice default plots.

% FS = 16;
FS = 20;
LW = 3;

%Default Figure properties
set(groot,'defaultFigureWindowStyle','docked');
set(groot,'defaultFigureColor','white');

%Default Text properties
set(groot,'defaultTextInterpreter','latex');
set(groot,'defaultTextFontSize',FS);
set(groot,'defaultTextLineWidth',0.5);

% these three affect titles, xlabels, etc. too...
set(groot,'defaultTextEdgeColor','none');
set(groot,'defaultTextBackgroundColor','none');

%Default Axes properties
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultAxesFontSize',FS);
set(groot,'defaultAxesXGrid','on');
set(groot,'defaultAxesYGrid','on');
set(groot,'defaultAxesXMinorTick','on');
set(groot,'defaultAxesYMinorTick','on');

%Default Line properties
set(groot,'defaultLineLineWidth',LW);

%Default Legend properties
set(groot,'defaultLegendInterpreter','latex');
set(groot,'defaultLegendFontSize',FS);

%Default Colorbar properties
set(groot,'defaultColorbarTickLabelInterpreter','latex');
set(groot,'defaultColorbarFontSize',FS);

end

