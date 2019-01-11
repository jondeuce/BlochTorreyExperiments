function [ Results ] = BOLDCurve(Results, EchoTimes, dt, Y0, Y, Hct, Dcoeff, B0, AlphaRange, Geom, MaskType, type, stepper)

% Angle data
NumAngles   = numel(AlphaRange);
ResultsArgs = getargs(Results);

% Create kernelstepper before angle loop, as this way the k-space diffusion
% kernel is precomputed only once for all angles alpha
switch upper(stepper)
    case 'BTSPLITSTEPPER'
        Gamma = []; dGamma = {};
        kernelstepper = SplittingMethods.BTSplitStepper(...
            dt, Dcoeff, Gamma, dGamma, Geom.GridSize, Geom.VoxelSize, ...
            'NReps', 1, 'Order', 2);
        getstepper = @(gamma) precomputeExpDecays(kernelstepper, gamma);
    case 'EXPMVSTEPPER'
        getstepper = @(gamma) precompute(...
            ExpmvStepper(dt, ...
                setbuffer( ...
                    BlochTorreyOp(gamma, Dcoeff, Geom.GridSize, Geom.VoxelSize, false, GetMask(Geom, MaskType)), ...
                    BlochTorreyOp.DiagState ...
                    ), ...
                Geom.GridSize, Geom.VoxelSize, ...
                'type', 'GRE', 'prec', 'half', ...%1e-6, ...
                'prnt', true, 'forcesparse', false, 'shift', true, ...
                'bal', false, 'full_term', false), ...
            gamma);
end

for ii = 1:NumAngles
    
    alpha_loop_time = tic;
    alpha = AlphaRange(ii);
    
    GammaSettingsY0 = Geometry.ComplexDecaySettings('Angle_Deg', alpha, 'B0', B0, 'Y', Y0, 'Hct', Hct);
    GammaSettingsY  = Geometry.ComplexDecaySettings('Angle_Deg', alpha, 'B0', B0, 'Y', Y,  'Hct', Hct);
    
    % ---- Baseline Signal: BloodOxygenation = Y0 ---- %
    t_Base  =  tic;
    
    V = getstepper( CalculateComplexDecay( GammaSettingsY0, Geom ) );
    [ Signal_Baseline ] = PropBOLDSignal( V, EchoTimes, type );
    Results = push( Results, Signal_Baseline, [], EchoTimes, deg2rad(alpha), ResultsArgs{3:end} );
    
    try
        save([datestr(now,30), '__IntermediateResults'], 'Results', '-v7');
    catch me
        warning(me.message)
    end
    display_toc_time( toc(t_Base), sprintf( 'Angle %2d/%2d, %5.2f%s, Baseline ', ii, NumAngles, alpha, '°' ) );
    
    % ---- Activated Signal: BloodOxygenation = Y ---- %
    t_Base  =  tic;
    
    V = getstepper( CalculateComplexDecay( GammaSettingsY, Geom ) );
    [ Signal_Activated ] = PropBOLDSignal( V, EchoTimes, type );
    Results = push( Results, [], Signal_Activated, EchoTimes, deg2rad(alpha), ResultsArgs{3:end} );
    
    try
        save([datestr(now,30), '__IntermediateResults'], 'Results', '-v7');
    catch me
        warning(me.message)
    end
    display_toc_time( toc(t_Base), sprintf( 'Angle %2d/%2d, %5.2f%s, Activated', ii, NumAngles, alpha, '°' ) );
    
    % ---- Total Time ---- %
    str = sprintf('Total time for alpha = %.2f',alpha);
    display_toc_time(toc(alpha_loop_time), str, [true,true]);
end

end

function [ Signal ] = PropBOLDSignal( V, EchoTimes, type )

ScaleSum = prod(V.VoxelSize) / prod(V.GridSize);
TE = EchoTimes(:).';
dt = V.TimeStep;

is_approx_eq = @(x,y) max(abs(x(:)-y(:))) <= 5*eps(max(max(abs(x(:))), max(abs(y(:)))));
switch upper(type)
    case 'GRE'
        if ~is_approx_eq(dt*round(TE/dt), TE)
            error('Each echotime must be disible by the timestep for GRE');
        end
    case 'SE'
        if ~is_approx_eq(2*dt*round(TE/(2*dt)), TE)
            error('Each echotime must be disible by twice the timestep for SE');
        end
end

addStartPoint = (TE(1)==0.0); % Can add back TE = 0.0 solution later
TE = TE(1+addStartPoint:end); % Don't simulate TE = 0.0
NumEchoTimes = numel( TE );

GRE_Steps = round( TE/dt );
GRE_Steps = [GRE_Steps(1), diff(GRE_Steps)];
SE_SecondHalfSteps = round( (TE/dt)/2 );
SE_FirstHalfSteps  = [SE_SecondHalfSteps(1), diff(SE_SecondHalfSteps)];

Time0 = 0;
M0 = double(1i);
Signal0 = M0 * prod(V.VoxelSize);
Signal = cell(NumEchoTimes,1);
for ll = 1:NumEchoTimes; Signal{ll} = [Time0, Signal0]; end

% ScaleSum converts from units of voxels^3 to um^3
IntegrateMagnetization = @(M) ScaleSum * sum(sum(sum(M,1),2),3);

% Initialize current magnetization
Mcurr = M0 * ones( V.GridSize );

for kk = 1:NumEchoTimes
    
    looptime  =  tic;

    switch upper(type)
        case 'GRE'
            
            for jj = 1:GRE_Steps(kk)
                
                [Mcurr,~,~,V] = step(V, Mcurr);
                for ll = kk:NumEchoTimes
                    Signal{ll} = [Signal{ll}; [Signal{ll}(end,1)+dt, IntegrateMagnetization(Mcurr)]];
                end
                
            end
            
        case 'SE'
            
            for jj = 1:SE_FirstHalfSteps(kk)
                
                [Mcurr,~,~,V] = step(V, Mcurr);
                for ll = kk:NumEchoTimes
                    Signal{ll} = [Signal{ll}; [Signal{ll}(end,1)+dt, IntegrateMagnetization(Mcurr)]];
                end
                
            end
            
            M  =  Mcurr;
            M  =  conj(M);
            
            for jj = 1:SE_SecondHalfSteps(kk)
                
                [M,~,~,V] = step(V, M);
                Signal{kk}  =  [Signal{kk}; [Signal{kk}(end,1)+dt, IntegrateMagnetization(M)]];
                
            end
    end
    
    str = sprintf( 'TE = %5.1fms', 1000*Signal{kk}(end,1));
    display_toc_time(toc(looptime),str);
    
end

if addStartPoint
    % Add initial signal to the front of subsequent simulated signals
    Signal = [Signal{1}(1,:); Signal];
end

end
