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
    fig = sliceplot(real(mid(args.Gamma)));
    if args.SavePlots; saveplot(fig, 'R2'); close(fig); end

    fig = sliceplot(imag(mid(args.Gamma)), 'contrast', infnorm(imag(mid(args.Gamma))) * [-1,1]);
    if args.SavePlots; saveplot(fig, 'dOmega'); close(fig); end
end

if ~isempty(args.Magnetization)
    fig = sliceplot(abs(mid(args.Magnetization)), 'contrast', [0,1]);
    if args.SavePlots; saveplot(fig, 'Magnitude'); close(fig); end

    fig = sliceplot(detrend(angle(mid(args.Magnetization))), 'contrast', [-pi/4,pi/4]);
    if args.SavePlots; saveplot(fig, 'Phase'); close(fig); end
end

if ~isempty(args.DiffusionMap)
    fig = sliceplot(mid(args.DiffusionMap));
    if args.SavePlots; saveplot(fig, 'DiffusionMap'); close(fig); end
end

if ~isempty(args.Time) && ~isempty(args.Signal)
    fig = figure;
    plot(1000 * args.Time, abs(args.Signal/args.Signal(1)), '-', 'marker', '.', 'markersize', 15, 'markeredgecolor', 'k');
    ylabel('Signal Magnitude [a.u.]');
    xlabel('Time [ms]');
    if args.SavePlots; saveplot(fig, 'SignalMagnitude'); close(fig); end

    fig = figure;
    plot(1000 * args.Time, detrend(angle(args.Signal)), '-', 'marker', '.', 'markersize', 15, 'markeredgecolor', 'k');
    ylabel('Signal Phase [rad]');
    xlabel('Time [ms]');
    if args.SavePlots; saveplot(fig, 'SignalPhase'); close(fig); end
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
fig = Plotter(x, 'color', 'plasma', varargin{:});
fig = fig.hfig{1};
figure(fig)
set(fig, 'color', 'w')
colorbar(gca(fig))
end

function ph = detrend(ph)
% Removes square wave pattern from 180 pulses
ph = (ph > 0) .* (ph - pi/2) - (ph <= 0) .* (ph + pi/2);
end
