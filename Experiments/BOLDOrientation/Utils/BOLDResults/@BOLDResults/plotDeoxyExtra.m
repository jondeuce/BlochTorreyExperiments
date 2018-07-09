function [ varargout ] = plotDeoxyExtra( R, varargin )
%PLOT

p = getInputParser;
parse(p,varargin{:})
opts = p.Results;

[echotimes, alphas, deoxysignals, deoxysignalsIntra, deoxysignalsExtra, deoxysignalsVRS, ...
                  oxysignals, oxysignalsIntra, oxysignalsExtra, oxysignalsVRS] = getboldplotargs(R,opts);
for idx = 1:numel(deoxysignalsExtra)
    [~,~,fig(idx)] = plotBOLDSignal(R,echotimes,alphas,deoxysignalsExtra{idx},opts);
end

switch nargout
    case 1
        varargout{1} = fig;
    case 3
        [varargout{1:3}] = deal(echotimes, alphas, deoxysignalsExtra);
end

end

function [hbold,leg,fig] = plotBOLDSignal(R,echotimes,alphas,deoxysignalsExtra,opts)

switch upper(opts.scalefactor)
    case 'NORMALIZE'
        opts.scalefactor = 1/max(deoxysignalsExtra(:));
    case 'RELATIVE'
        opts.scalefactor = 1/prod(R.MetaData.SimSettings.VoxelSize);
end
deoxysignalsExtra = opts.scalefactor * deoxysignalsExtra;

cmap = jet(numel(alphas));

switch upper(opts.fig)
    case ''; fig = figure;
    case 'CURRENT'; fig = gcf;
end

hold on
hx = plot(gca, echotimes, deoxysignalsExtra, 'x', 'markersize', 8);

if opts.interp
    switch upper(opts.interptype)
        case 'SPLINE'
            coeffs = spline(echotimes.',deoxysignalsExtra.');
            %[TEinterps,boldinterps] = fplot(@(t)ppval(coeffs,t(:).').',[echotimes(1),echotimes(end)]);
            TEinterps = linspace(echotimes(1),echotimes(end),200);
            boldinterps = ppval(coeffs,TEinterps);
            figure(fig), hbold = plot(TEinterps, boldinterps);
            ymax = max( max(deoxysignalsExtra(:)), max(boldinterps(:)) );
            ymin = min( min(deoxysignalsExtra(:)), min(boldinterps(:)) );
        case 'CHEBFUN'
            boldinterps = chebfun(deoxysignalsExtra,[echotimes(1),echotimes(end)],'equi');
            figure(fig), hbold = plot(boldinterps);
            ymax = max( max(deoxysignalsExtra(:)), max(max(boldinterps)) );
            ymin = min( min(deoxysignalsExtra(:)), min(min(boldinterps)) );
    end
else
    figure(fig), hbold = plot(echotimes, deoxysignalsExtra);
    ymax = max(deoxysignalsExtra(:));
    ymin = min(deoxysignalsExtra(:));
end
ylim([ymin,ymax]);

for ii = 1:numel(alphas)
    set(hx(ii),'color',cmap(ii,:));
    set(hbold(ii),'color',cmap(ii,:));
end

if strcmpi(opts.angleunits,'rad'); legunit = '\,\mathrm{rad}';
else legunit = '^\circ';
end
labels = strcat( repmat({'$\alpha = '},numel(alphas),1), num2str(alphas,3), legunit, '$' );

leg = [];
if strcmpi(opts.legend,'on')
    leg = legend(hbold,labels{:});
    set(leg,'location',opts.legendlocation,'interpreter','latex');
end

if ~isempty(opts.xlabel); xlabel(opts.xlabel); end
if ~isempty(opts.ylabel); ylabel(opts.ylabel); end
if ~isempty(opts.title); title(opts.title); end

xmin = max(opts.minTE,min(echotimes)); % for when opts.minTE = -inf
xmax = min(opts.maxTE,max(echotimes)); % for when opts.maxTE = +inf
xlim([xmin,xmax]);

end

function [echotimes, alphas, deoxysignals, deoxysignalsIntra, deoxysignalsExtra, deoxysignalsVRS, ...
                  oxysignals, oxysignalsIntra, oxysignalsExtra, oxysignalsVRS] = getboldplotargs(R,opts)

% Convert to standard for BOLDResults
echotimes = convertEchotimes(opts.echotimes,opts.timeunits,'s');
alphas = convertAlphas(opts.alphas,opts.angleunits,'rad');

boldargs = {echotimes,alphas};
[echotimes, alphas, ...
 boldsignals, boldsignalsIntra, boldsignalsExtra, boldsignalsVRS, ...
 deoxysignals, deoxysignalsIntra, deoxysignalsExtra, deoxysignalsVRS, ...
 oxysignals, oxysignalsIntra, oxysignalsExtra, oxysignalsVRS] = getBOLDSignals(R,boldargs{:});

% Convert to desired units for plotting
echotimes = convertEchotimes(echotimes,'s',opts.timeunits);
alphas = convertAlphas(alphas,'rad',opts.angleunits);

% Get only desired TEs, alphas, and signals
if any(~isinf([opts.minTE,opts.maxTE,opts.minAlpha,opts.maxAlpha]))
    TEidx = opts.minTE <= echotimes & echotimes <= opts.maxTE;
    alphaidx = opts.minAlpha <= alphas & alphas <= opts.maxAlpha;
    echotimes = echotimes(TEidx);
    alphas = alphas(alphaidx);
    for idx = 1:numel(deoxysignalsExtra)
        deoxysignalsExtra{idx} = deoxysignalsExtra{idx}(TEidx, alphaidx);
    end
end

end

function TE = convertEchotimes(echotimes,unitfrom,unitto)

TE = echotimes;

%Convert echotimes to seconds
switch upper(unitfrom)
    case 'MS'
        TE = TE / 1000;
end

%Convert seconds to desired
switch upper(unitto)
    case 'MS'
        TE = TE * 1000;
end

end

function angles = convertAlphas(alphas,unitfrom,unitto)

angles = alphas;

%Convert alphas to rads
switch upper(unitfrom)
    case 'DEG'
        angles = (pi/180) * angles;
end

%Convert angles to desired
switch upper(unitto)
    case 'DEG'
        angles = (180/pi) * angles;
end

end

function p = getInputParser

p = inputParser;
p.FunctionName = 'plot';

VA = @(varargin) validateattributes(varargin{:});
VS = @(varargin) validatestring(varargin{:});

addParameter(p,'type','BOLD',@(x)VA(x,{'char'},{'nonempty'}));
addParameter(p,'scalefactor','normalize',@(x)isnumeric(x) || ischar(x));

addParameter(p,'xlabel','TE [ms]',@(x)VA(x,{'char'},{'nonempty'}));
addParameter(p,'ylabel','BOLD Signal',@(x)VA(x,{'char'},{'nonempty'}));
addParameter(p,'title','',@(x)VA(x,{'char'},{'nonempty'}))

expectedfigarg = {'current'};
addParameter(p,'fig','',@(x) any(VS(x,expectedfigarg)));

expectedfigarg = {'on','off'};
addParameter(p,'legend','on',@(x) any(VS(x,expectedfigarg)));
addParameter(p,'legendlocation','best',@(x)VA(x,{'char'},{'nonempty'}));

expectedangleunits = {'deg','rad'};
addParameter(p,'angleunits','deg',@(x) any(VS(x,expectedangleunits)));

expectedtimeunits = {'s','ms'};
addParameter(p,'timeunits','ms',@(x) any(VS(x,expectedtimeunits)));

addParameter(p,'alphas',[],@(x)VA(x,{'numeric'},{'nonempty'}));
addParameter(p,'echotimes',[],@(x)VA(x,{'numeric'},{'nonempty'}));

addParameter(p,'minAlpha',-inf,@(x)VA(x,{'numeric'},{'nonempty'}));
addParameter(p,'maxAlpha',+inf,@(x)VA(x,{'numeric'},{'nonempty'}));
addParameter(p,'minTE',-inf,@(x)VA(x,{'numeric'},{'nonempty'}));
addParameter(p,'maxTE',+inf,@(x)VA(x,{'numeric'},{'nonempty'}));

addParameter(p,'interp',true,@(x)VA(x,{'logical'},{'scalar'}));
expectedinterptype = {'spline','chebfun'};
addParameter(p,'interptype','spline',@(x) any(VS(x,expectedinterptype)));

end

