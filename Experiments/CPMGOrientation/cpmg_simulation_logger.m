function cpmg_simulation_logger(varargin)
%CPMG_SIMULATION_LOGGER

args = parseinputs(varargin{:});

suf = '';
if ~isempty(args.Time)
    suf = sprintf('_step_%4.4d_time_%4.4f_ms', numel(args.Time)-1, 1000 * args.Time(end));
end

mid = @(x) x(:,:,floor(end/2));
saveplot = @(fig, s) saveas(fig, [s, suf, '.png']);

if ~isempty(args.Gamma)
    fig = sliceplot(real(mid(args.Gamma)), 'title', 'Local Relaxation Rate [Hz]');
    if args.SavePlots; saveplot(fig, 'R2'); close(fig); end

    fig = sliceplot(imag(mid(args.Gamma)), 'contrast', 'symmetric', 'title', 'Local Field Inhomogeneities [rad/s]');
    if args.SavePlots; saveplot(fig, 'dOmega'); close(fig); end
end

if ~isempty(args.Magnetization)
    fig = sliceplot(abs(mid(args.Magnetization)), 'contrast', [0,1], 'title', 'Magnetization Magnitude [a.u.]');
    if args.SavePlots; saveplot(fig, 'Magnitude'); close(fig); end

    fig = sliceplot(detrend(angle(mid(args.Magnetization))), 'contrast', 'symmetric', 'title', 'Magnetization Phase (de-trended) [rad]');
    if args.SavePlots; saveplot(fig, 'Phase'); close(fig); end
end

if ~isempty(args.DiffusionMap)
    fig = sliceplot(mid(args.DiffusionMap), 'title', 'Local Diffusivity D(x) [um2/s]');
    if args.SavePlots; saveplot(fig, 'Diffusivity'); close(fig); end
end

if ~isempty(args.Time) && ~isempty(args.Signal)
    fig = figure;
    ax = gca(fig);

    subplot(1,2,1);
    plot(1000 * args.Time, abs(args.Signal/args.Signal(1)), '-', 'marker', '.', 'markersize', 15, 'markeredgecolor', 'k');
    ylabel('Signal Magnitude [a.u.]');
    xlabel('Time [ms]');
    xlim(1000 * vec(args.Time([1,end])).')

    subplot(1,2,2);
    plot(1000 * args.Time, detrend(angle(args.Signal)), '-', 'marker', '.', 'markersize', 15, 'markeredgecolor', 'k');
    ylabel('Signal Phase (de-trended) [rad]');
    xlabel('Time [ms]');
    xlim(1000 * vec(args.Time([1,end])).')

    if args.SavePlots; saveplot(fig, 'Signal'); close(fig); end
end

end

function args = parseinputs(varargin)

p = inputParser;
p.FunctionName = 'cpmg_simulation_logger';

addParameter(p, 'SavePlots', false);
addParameter(p, 'Time', []);
addParameter(p, 'Signal', []);
addParameter(p, 'Magnetization', []);
addParameter(p, 'Gamma', []);
addParameter(p, 'DiffusionMap', []);

parse(p, varargin{:});
args = p.Results;

end

function fig = sliceplot(x, varargin)

p = inputParser;
addParameter(p, 'colormap', 'plasma');
addParameter(p, 'contrast', []);
addParameter(p, 'title', '');
parse(p, varargin{:});
args = p.Results;

fig = figure;
ax = gca(fig);
hold on; grid off; axis off; axis image;

clim = {};
if ischar(args.contrast) && strcmpi(args.contrast, 'symmetric')
    clim = {infnorm(abs(x)) * [-1,1]};
elseif isnumeric(args.contrast) && numel(args.contrast) == 2
    clim = {args.contrast(:).'};
end
imagesc(x, clim{:});

set(fig, 'color', 'w')
colormap(ax, args.colormap)
colorbar(ax)
if ~isempty(args.title); title(ax, args.title); end

end

function ph = detrend(ph)
% Subtracts off the square wave pattern in the phase induced by the 180 pulses
ph = (ph > 0) .* (ph - pi/2) - (ph <= 0) .* (ph + pi/2);
end
