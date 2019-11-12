function [maps,distributions] = T2map_SEcorr(image,varargin)
%
% [maps,distributions] = T2map_SEcorr(image,...)
%
% Description:
%   Uses NNLS to compute T2 distributions in the presence of stimulated
%   echos by optimizing the refocusing pulse flip angle.  Records parameter
%   maps and T2 distributions for further partitioning.
%
% Inputs:
%   image: 4-D array with intensity data as (row,column,slice,echo)
%   ...: A series of optional Property/Value pairs to modify settings.
%     Defaults are given in brackets:
%       'TE': Interecho spacing, usually set to one number, but may also
%             be set to 'variable' for sequences with 2 interecho spacings
%             (see 'vTEparam'). (0.01)
%       'vTEparam': [TE1,TE2,number of echoes at TE1]. Only applied when
%                   'TE' set to 'variable'. TE2 must be and integer
%                   multiple of TE1. ([0.01,0.05,16])
%       'nT2': Number of T2 times to use (40)
%       'T2Range': Min and Max T2 values ([0.015,2.000])
%       'T1': Assumed value of T1 (1)
%       'Threshold': First echo intensity cutoff for empty voxels (200)
%       'Reg': Regularization routine to use, options are:
%              'no': do not regularize the solution
%              'chi2': use Chi2Factor based regularization (default)
%              'lcurve': use L-Curve based regularization
%       'Chi2Factor': Constraint on chi^2 used for regularization (Reg must
%                     be set to 'chi2'!) (1.02)
%       'RefCon': Refocusing Pulse Control Angle (180)
%       'MinRefAngle': Minimum refocusing angle for EPG optimization (50)
%       'nAngles': Number of angles used in EPG optimization (8)
%       'SetFlipAngle': Instead of optimizing flip angle, uses this flip
%                       angle for all voxels (not set)
%       'nCores': Number of processor cores to use (6)
%       'Save_regparam': yes/no option to include the regularization
%                        paramter mu and the resulting chi^2 factor as
%                        two outputs within the maps structure (mu=NaN and
%                        chi2factor=1 if Reg=no) ('no')
%       'Save_NNLS_basis': yes/no option to include a 5-D matrix of NNLS
%                          basis matrices as another output within the maps
%                          structure ('no')
%       'Waitbar': yes/no option determining whether a progress bar is
%                  generated.  Selecting 'no' will also suppress any
%                  mesages printed to the command window. ('yes')
%
% Ouputs:
%   maps: Structure containing 3D maps of the following parameters
%       -gdn, general density
%       -ggm, general geometric mean
%       -gva, general variance
%       -FNR, fit to noise ratio (gdn/stdev(residuals))
%       -alpha, refocusing pulse flip angle
%   distributions: 4-D matrix containing T2 distributions.
%
% External Calls:
%   EPGdecaycurve.m
%   EPGdecaycurve_vTE.m
%   lsqnonneg_reg.m
%   lsqnonneg_lcurve.m
%
% Created by Thomas Prasloski
% email: tprasloski@gmail.com
% Ver. 3.3, August 2013

%==========================================================================
% Parse inputs and apply default values when necessary
%==========================================================================
% Make image double
image = double(image);
% Create input parser object
p = inputParser;
% Define all input values
p.addRequired('image',@(x)isa(x,'double') && ndims(x) == 4);
p.addParamValue('TE',0.010,@(x)(isnumeric(x) && isscalar(x) && x >= 0.0001 && x <= 1)||strcmp(x,'variable'));
p.addParamValue('vTEparam',[0.01,0.05,16],@(x)isnumeric(x) && length(x) == 3 && x(2) > x(1) && mod(x(2)/x(1),1) == 0 && x(3) < size(image,4))
p.addParamValue('T1',1,@(x)isnumeric(x) && isscalar(x) && x >= 10 && x <= 0.001);
p.addParamValue('RefCon',180,@(x)isnumeric(x) && isscalar(x) && x <= 180 && x >= 1);
p.addParamValue('Threshold',200,@(x)isnumeric(x) && isscalar(x));
p.addParamValue('Chi2Factor',1.02,@(x)isnumeric(x)  && isscalar(x) && x > 1);
p.addParamValue('nT2',40,@(x)isnumeric(x) && isscalar(x) && x >= 10 && x <= 120);
p.addParamValue('T2Range',[0.0150,2.0000],@(x)isnumeric(x) && length(x) == 2 && x(2) > x(1) && x(1) >= 0.001 && x(2) <= 10);
p.addParamValue('MinRefAngle',50,@(x)isnumeric(x) && isscalar(x) && x > 1 && x < 180);
p.addParamValue('nAngles',8,@(x)isnumeric(x) && isscalar(x) && x > 1);
p.addParamValue('Reg','chi2',@(x)any(strcmp(x,{'no','chi2','lcurve'})));
p.addParamValue('SetFlipAngle',0,@(x)(isnumeric(x) && isscalar(x)));
p.addParamValue('nCores',6,@(x)isnumeric(x) && isscalar(x) && x >= 1 && x <= 8);
p.addParamValue('Save_regparam','no',@(x)any(strcmp(x,{'yes','no'})));
p.addParamValue('Save_NNLS_basis','no',@(x)any(strcmp(x,{'yes','no'})));
p.addParamValue('Waitbar','yes',@(x)any(strcmp(x,{'yes','no'})));
% Parse inputs (MATLAB will throw an error here if any variables fail validation)
p.parse(image,varargin{:});
% Define all variables from the inputParser Results
TE = p.Results.TE;
if strcmp(TE,'variable')
    TE1 = p.Results.vTEparam(1);
    TE2 = p.Results.vTEparam(2);
    nTE1 = p.Results.vTEparam(3);
end
T1 = p.Results.T1;
RefCon = p.Results.RefCon;
Threshold = p.Results.Threshold;
Chi2Factor = p.Results.Chi2Factor;
nT2 = p.Results.nT2;
T2Range = p.Results.T2Range;
minangle = p.Results.MinRefAngle;
nangles = p.Results.nAngles;
reg = p.Results.Reg;
faset = p.Results.SetFlipAngle;
nCores = p.Results.nCores;
savereg = strcmp(p.Results.Save_regparam,'yes');
saveNNLS = strcmp(p.Results.Save_NNLS_basis,'yes');
waitbar = strcmp(p.Results.Waitbar,'yes');

%==========================================================================
% Initialize all the data
%==========================================================================
% Start the clock
tstart = tic;
% Find size of the data
[nrows,ncols,nslices,nechs] = size(image);
% Initialize map matrices
gdnmap = nan*ones(nrows,ncols,nslices);
ggmmap = nan*ones(nrows,ncols,nslices);
gvamap = nan*ones(nrows,ncols,nslices);
SNRmap = nan*ones(nrows,ncols,nslices);
FNRmap = nan*ones(nrows,ncols,nslices);
alphamap = nan*ones(nrows,ncols,nslices);
distributions = nan*ones(nrows,ncols,nslices,nT2);
mumap = nan*ones(nrows,ncols,nslices);
chi2map = nan*ones(nrows,ncols,nslices);
if saveNNLS
    decay_basis = nan*ones(nrows,ncols,nslices,nechs,nT2);
end
if waitbar
    % Estimate completion time
    num = sum(sum(sum(image(:,:,:,1) >= Threshold)));
    est = (num*0.0117)*(nechs/32)*(nT2/40)*(nangles/8)*(8/nCores);
    if faset ~= 0
        est = est/5.2;
    end
    fprintf('ESTIMATED completion time is %2.0f hours, %2.0f minutes...\n',floor(est/3600),(est/3600-floor(est/3600))*60)
end
%==========================================================================
% Find the basis matrices for each flip angle
%==========================================================================
% Initialize parameters and variable for angle optimization
T2_times = logspace(log10(T2Range(1)),log10(T2Range(2)),nT2);
if faset == 0
    flip_angles = linspace(minangle,180,nangles);
    % basis_angles is a 1xnangles cell array that will contain the decay bases of each angle
    basis_angles = cell(nangles);
    % Loop to compute each basis and assign them to a cell in the array
    basis_decay = zeros(nechs,nT2);
    for a = 1:nangles
        for x = 1:nT2
            if strcmp(TE,'variable')
                echo_amp = EPGdecaycurve_vTE(nechs,flip_angles(a),TE1,TE2,nTE1,T2_times(x),T1,RefCon);
            else
                echo_amp = EPGdecaycurve(nechs,flip_angles(a),TE,T2_times(x),T1,RefCon);
            end
            basis_decay(:,x) = echo_amp';
        end
        basis_angles{a} = basis_decay;
    end
    basis_decay_faset = [];  %ignore
else
    basis_angles = [];  %ignore
    flip_angles = [];  %ignore
    basis_decay_faset = zeros(nechs,nT2);
    for x = 1:nT2
        if strcmp(TE,'variable')
            echo_amp = EPGdecaycurve_vTE(nechs,faset,TE1,TE2,nTE1,T2_times(x),T1,RefCon);
        else
            echo_amp = EPGdecaycurve(nechs,faset,TE,T2_times(x),T1,RefCon);
        end
        basis_decay_faset(:,x) = echo_amp';
    end
end
if ~strcmp(TE,'variable')
    TE1 = [];  %ignore
    TE2 = [];  %ignore
    nTE1 = [];  %ignore
end
%==========================================================================
% Process all pixels
%==========================================================================
% Main triple for-loop to run through each pixel in the image

USE_PARALLEL = false;
poolopenflag = ~isempty(gcp('nocreate'));

if USE_PARALLEL && ~poolopenflag
    parpool('local');
    poolopenflag = true;
elseif ~USE_PARALLEL && poolopenflag
    delete(gcp('nocreate')); %matlabpool close
    poolopenflag = false;
end

currpool = gcp('nocreate');
numworkers = 1;
if ~isempty(currpool)
    numworkers = currpool.NumWorkers;
end

try
    
    if waitbar
        if USE_PARALLEL
            pctRunOnAll javaaddpath /data/workgroup/matlab/testing_functions/SEcorr/java
        else
            javaaddpath /data/workgroup/matlab/testing_functions/SEcorr/java
        end
        
        progressStepSize = 1;
        ppm = ParforProgMon('T2map_SEcorr: ', nrows, progressStepSize, 400, 80);
    else
        ppm = [];
    end
        
    %parfor row = 1:nrows
    for row = 1:nrows
        gdn = nan*ones(ncols,nslices);
        ggm = nan*ones(ncols,nslices);
        gva = nan*ones(ncols,nslices);
        SNR = nan*ones(ncols,nslices);
        FNR = nan*ones(ncols,nslices);
        alpha = nan*ones(ncols,nslices);
        chi2_alpha = nan*ones(1,nangles);
        dists = nan*ones(ncols,nslices,nT2);
        mus = nan*ones(ncols,nslices);
        chi2s = nan*ones(ncols,nslices);
        T2_dis = zeros(nT2,1);
        mu = 0;
        chi2 = 0;
        if saveNNLS
            basis_matrices = nan*ones(ncols,nslices,nechs,nT2);
        else
            basis_matrices = [];
        end
        rowloopstart = tic;
        rowloopcount = 0;
        for col = 1:ncols
            tfinish_est = nrows * ncols * nslices * (toc(rowloopstart) / rowloopcount) / numworkers;
            fprintf('Starting row %3d/%3d, column %3d/%3d -- Time: %2.0f hours, %2.0f minutes -- Estimated Finish: %2.0f hours, %2.0f minutes\n',...
               row,nrows,col,ncols,floor(toc(tstart)/3600),(toc(tstart)/3600-floor(toc(tstart)/3600))*60,floor(tfinish_est/3600),(tfinish_est/3600-floor(tfinish_est/3600))*60);
            for slice = 1:nslices
                % Conditional loop to reject low signal pixels
                if image(row,col,slice,1) >= Threshold
                    % Extract decay curve from the pixel
                    decay_data = squeeze(image(row,col,slice,1:nechs));
                    if faset == 0
                        %======================================================
                        % Find optimum flip angle
                        %======================================================
                        % Fit each basis and find chi-squared
                        for a = 1:nangles
                            T2_dis_ls = lsqnonneg(basis_angles{a},decay_data);
                            decay_pred = basis_angles{a}*T2_dis_ls;
                            chi2_alpha(a) = sum((decay_data-decay_pred).^2);
                        end
                        % Find the minimum chi-squared and the corresponding angle
                        alpha_spline = flip_angles(1):0.001:flip_angles(end);
                        chi2_spline = interp1(flip_angles,chi2_alpha,alpha_spline,'spline');
                        [~,index] = min(chi2_spline);
                        alpha(col,slice) = alpha_spline(index);
                        %======================================================
                        % Fit basis matrix using alpha
                        %======================================================
                        basis_decay = zeros(nechs,nT2);
                        % Compute the NNLS basis over T2 space
                        for x = 1:nT2
                            if strcmp(TE,'variable')
                                echo_amp = EPGdecaycurve_vTE(nechs,alpha(col,slice),TE1,TE2,nTE1,T2_times(x),T1,RefCon);
                            else
                                echo_amp = EPGdecaycurve(nechs,alpha(col,slice),TE,T2_times(x),T1,RefCon);
                            end
                            basis_decay(:,x) = echo_amp';
                        end
                    else
                        alpha(col,slice) = faset;
                        basis_decay = basis_decay_faset;
                    end
                    if saveNNLS
                        basis_matrices(col,slice,:,:) = basis_decay;
                    end
                    %==========================================================
                    % Calculate T2 distribution and global parameters
                    %==========================================================
                    % Find distribution depending on regularization routine
                    switch reg
                        case 'no'
                            % Fit T2 distribution using unregularized NNLS
                            T2_dis = lsqnonneg(basis_decay,decay_data);
                            mu = nan;
                            chi2 = 1;
                        case 'chi2'
                            % Fit T2 distribution using chi2 based regularized NNLS
                            [T2_dis,mu,chi2] = lsqnonneg_reg(basis_decay,decay_data,Chi2Factor);
                        case 'lcurve'
                            % Fit T2 distribution using lcurve based regularization
                            [T2_dis,mu,chi2] = lsqnonneg_lcurve(basis_decay,decay_data);
                    end
                    dists(col,slice,:) = T2_dis;
                    mus(col,slice) = mu;
                    chi2s(col,slice) = chi2;
                    % Compute parameters of distribution
                    gdn(col,slice) = sum(T2_dis);
                    ggm(col,slice) = exp(dot(T2_dis,log(T2_times))/sum(T2_dis));
                    gva(col,slice) = exp(sum((log(T2_times)-log(ggm(col,slice))).^2.*T2_dis')./sum(T2_dis)) - 1;
                    decay_calc = basis_decay*T2_dis;
                    residuals = decay_calc-decay_data;
                    FNR(col,slice) = sum(T2_dis)/sqrt(var(residuals));
                    SNR(col,slice) = max(decay_data)/sqrt(var(residuals));
                end
                rowloopcount = rowloopcount + 1;
            end
        end
        % Record temporary maps into 3D outputs
        gdnmap(row,:,:) = gdn;
        ggmmap(row,:,:) = ggm;
        gvamap(row,:,:) = gva;
        SNRmap(row,:,:) = SNR;
        FNRmap(row,:,:) = FNR;
        alphamap(row,:,:) = alpha;
        distributions(row,:,:,:) = dists;
        mumap(row,:,:) = mus;
        chi2map(row,:,:) = chi2s;
        if saveNNLS
            decay_basis(row,:,:,:,:) = basis_matrices;
        end
        if waitbar
            % Increment counter for waitbar
            if mod(row,1) == 0
                ppm.increment();
            end
        end
    end
    
    if waitbar
        ppm.delete();
    end
    
catch err
    
    if poolopenflag == 1
        delete(gcp('nocreate')) %matlabpool close
    end
    rethrow(err)
    
end

if poolopenflag == 1
    delete(gcp('nocreate')) %matlabpool close
end

% Assign outputs
maps.gdn = gdnmap;
maps.ggm = ggmmap;
maps.gva = gvamap;
maps.alpha = alphamap;
maps.FNR = FNRmap;
if savereg
    maps.mu = mumap;
    maps.chi2factor = chi2map;
end
if saveNNLS
    maps.NNLS_basis = decay_basis;
end

if waitbar
    % Print message on finish with image name and total run time
    name = inputname(1);
    fprintf('%s completed in %2.0f hours, %2.0f minutes\n',name,floor(toc(tstart)/3600),(toc(tstart)/3600-floor(toc(tstart)/3600))*60)
end

end
