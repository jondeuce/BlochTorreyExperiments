function [dR2, ResultsStruct] = perforientation_lsqoptfun(params,xdata,dR2_Data,TE,VoxelSize,VoxelCenter,GridSize,B0,Dcoeff,Nsteps,Nmajor,Rminor_mu,Rminor_sig,Navgs,type,order,SaveFigs,SaveResults,DiaryFilename,geomseed)
%perforientation_lsqoptfun

if nargin < 20 || isempty(geomseed); geomseed = []; end
if nargin < 19 || isempty(DiaryFilename); DiaryFilename = []; end
if nargin < 18 || isempty(SaveResults); SaveResults = true; end
if nargin < 17 || isempty(SaveFigs); SaveFigs = true; end

CA = params(1);
iBVF = params(2);
aBVF = params(3);
BVF = iBVF + aBVF;
iRBVF = iBVF/BVF;
alpha_range = xdata;

new_geometry = @() Geometry.CylindricalVesselFilledVoxel( ...
    'BVF', BVF, 'iRBVF', iRBVF, ...
    'VoxelSize', VoxelSize, 'GridSize', GridSize, 'VoxelCenter', VoxelCenter, ...
    'Nmajor', Nmajor, 'Rminor_mu', Rminor_mu, 'Rminor_sig', Rminor_sig, ...
    'AllowMinorSelfIntersect', true, 'AllowMinorMajorIntersect', false, ...
    'PopulateIdx', true, 'seed', geomseed );

solver = @(G) SplittingMethods.PerfusionCurve( ...
    TE, Nsteps, Dcoeff, CA, B0, alpha_range, G, type, 'Order', order );

Geom = new_geometry();
AllGeoms = Compress(Geom);

%S_noCA and S_CA are arrays of size [numel(TimePts) x Nalphas x Navgs]
[~, S_noCA, S_CA, TimePts] = solver(Geom);

for ii = 2:Navgs
    Geom = new_geometry();
    AllGeoms = [AllGeoms; Compress(Geom)];
    [~, S_noCA__, S_CA__, TimePts__] = solver(Geom);
    
    TimePts = cat(3, TimePts, TimePts__);
    S_noCA  = cat(3, S_noCA, S_noCA__);
    S_CA    = cat(3, S_CA, S_CA__);
end

S_noCA_avg = mean(S_noCA(end,:,:),3); %row vector: [1 x Nalphas]
S_CA_avg   = mean(S_CA(end,:,:),3); %row vector: [1 x Nalphas]

%Need to squeeze after shift dim to handle cases of Navgs = 1
dR2_all = -1/TE .* squeeze( shiftdim( log(abs(S_CA(end,:,:)./S_noCA(end,:,:))) ) ).'; %array: [Navgs x Nalphas]
dR2 = -1/TE .* log( abs(S_CA_avg)./abs(S_noCA_avg) ); %row vector: [1 x Nalphas]

Results = struct( ...
    'params', params, ...
    'xdata', xdata, ...
    'dR2_Data', dR2_Data, ...
    'CA', CA, ...
    'iBVF', iBVF, ...
    'aBVF', aBVF, ...
    'alpha_range', alpha_range, ...
    'TimePts', TimePts, ...
    'dR2', dR2, ...
    'dR2_all', dR2_all, ...
    'S_CA', S_CA, ...
    'S_noCA', S_noCA, ...
    'Geometries', AllGeoms ...
    );

% ---- Plotting ---- %
try
    vec = @(x) x(:);
    %res = sqrt(sum((vec(dR2_Data)-vec(dR2(end,:))).^2)/numel(xdata));
    res = sum((vec(dR2_Data)-vec(dR2(end,:))).^2);
    avg_rminor = mean(vec([AllGeoms.Rminor_mu]));
    avg_rmajor = mean(vec([AllGeoms.Rmajor]));
    
    title_lines = {...
        sprintf('iBVF = %.4f%%, aBVF = %.4f%%, CA = %.4f, BVF = %.4f%%, iRBVF = %.2f%%', iBVF*100, aBVF*100, CA, BVF*100, iRBVF*100 ), ...
        sprintf('N = %d, Rmajor = %.2fum, Rminor = %.2fum, L2-residual = %.4f', Nmajor, avg_rmajor, avg_rminor, res) ...
        };
    title_lines = cellfun(@(s)strrep(s,'%','\%'),title_lines,'uniformoutput',false);
    title_str = [title_lines{1},', ',title_lines{2}];
    
    fname = strrep( title_str, ' ', '' );
    fname = strrep( fname, '\%', '' );
    fname = strrep( fname, '=', '-' );
    fname = strrep( fname, ',', '__' );
    fname = strrep( fname, '.', 'p' );
    fname = [datestr(now,30), '__', fname];

    fig = figure; set(gcf,'color','w'); hold on
    
    plot(alpha_range(:), dR2_all.', '-.');
    h = plot(alpha_range(:), [dR2_Data(:), dR2.'], '-', ...
        'marker', '+', 'linewidth', 4, 'markersize', 10);
    
    props = {'fontsize',14,'interpreter','latex'};
    title(title_lines, props{:});
    
    dR2str = '\Delta R_2'; if strcmpi(type,'GRE'); dR2str = [dR2str, '^*']; end
    xlabel('$\alpha$ [deg]', props{:});
    ylabel(['$',dR2str,'$ [Hz]'], props{:});
    
    leg = legend(h, {['$',dR2str,'$ Data'],'Simulated'});
    set(leg, 'location', 'best', props{:});
    
    drawnow
catch me
    warning('Unable to draw figures.\nError message: %s', me.message);
end

% ---- Save Figure ---- %
try
    if SaveFigs
        savefig(fig, fname);
        export_fig(fname, '-pdf', '-transparent');
    end
catch me
    warning('Unable to save figure.\nError message: %s', me.message);
end

% ---- Save Results ---- %
try
    if SaveResults
        save(fname,'Results')
    end
catch me
    warning('Unable to save results.\nError message: %s', me.message);
    save(datestr(now,30),'Results');
end

% ---- Save Diary ---- %
try
    if ~isempty(DiaryFilename)
        diary(DiaryFilename);
    end
catch me
    warning('Unable to save diary.\nError message: %s', me.message);
end

if nargout > 1
    ResultsStruct = Results;
end

end

