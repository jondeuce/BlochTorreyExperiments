function [maps] = T2part_SEcorr(T2distributions,varargin)
%
% [maps] = T2part_SEcorr(T2distributions,...)
%
% Description:
%   Analyzes T2 distributions produced by T2map_SEcorr to produce data maps
%   of a series of parameters.
%
% Inputs:
%   T2distributions: 4-D array with data as (row,column,slice,T2 Amplitude)
%   ...: A series of optional Property/Value pairs to modify settings.
%     Defaults are given in brackets:
%       'T2Range': Min and Max T2 values of distribution ([0.015,2.000])
%       'spwin': Min and Max of the short peak window ([0.015,0.040])
%       'mpwin': Min and Max of the middle peak window ([0.040,0.200])
%       'Sigmoid': Apply sigmoidal weighting to the upper limit of the 
%                  short peak window. Value is the delta-T2 parameter 
%                  (distance in seconds on either side of the spwin upper 
%                  limit where sigmoid curve reaches 10% and 90%). (Default
%                  is no sigmoid weighting)
%
% Ouputs:
%   maps: a structure containing 3D data maps as fields.  Field are:
%       -sfr (small pool (myelin water) fraction)
%       -sgm (small pool (myelin water) geometric mean)
%       -mfr (medium pool (intra/extra water) fraction)
%       -mgm (medium pool (intra/extra water) geometric mean)
%        
% External Calls:
%   none
%
% Created by Thomas Prasloski
% email: tprasloski@gmail.com
% Ver. 1.2, August 2012

tstart=tic;

p=inputParser;
p.addRequired('T2distributions',@(x)isnumeric(x) && ndims(x)==4);
p.addParamValue('T2Range',[0.015,2],@(x)isnumeric(x) && length(x)==2 && x(2)>x(1) && x(1)>=0.001 && x(2)<=10);
p.addParamValue('spwin',[0.015,0.040],@(x)isnumeric(x) && length(x)==2 && x(2)>x(1));
p.addParamValue('mpwin',[0.040,0.200],@(x)isnumeric(x) && length(x)==2 && x(2)>x(1));
p.addParamValue('Sigmoid',0,@(x)isnumeric(x) && isscalar(x));
p.parse(T2distributions,varargin{:});
T2Range=p.Results.T2Range;
spwin=p.Results.spwin;
mpwin=p.Results.mpwin;
sig=p.Results.Sigmoid;

[nrows,ncols,nslices,nT2]=size(T2distributions);
T2_times=logspace(log10(T2Range(1)),log10(T2Range(2)),nT2);
sp=T2_times>=spwin(1) & T2_times<=spwin(2);
mp=T2_times>=mpwin(1) & T2_times<=mpwin(2);

sfr=nan*ones(nrows,ncols,nslices);
sgm=nan*ones(nrows,ncols,nslices);
mfr=nan*ones(nrows,ncols,nslices);
mgm=nan*ones(nrows,ncols,nslices);

if sig
    T2_50perc=spwin(2);
    k=0.1;
    T2_kperc=sig;
    % Curve reaches 50% at T2_50perc and is (k and 1-k)*100 percent at T2_50perc+-T2_kperc  
    sigma=abs(T2_kperc/(sqrt(2)*erfinv(2*k-1)));
    weights=1-normcdf(T2_times,T2_50perc,sigma)';
end

%h=waitbar(0,'Initializing...','Name','Generating Data Maps...');
%tstart2=tic;

for row=1:nrows
    for col=1:ncols
        for slice=1:nslices
            dist=squeeze(T2distributions(row,col,slice,:));
            
            if sig
                % Use sigmoidal weighting function
                sfr(row,col,slice)=sum(dist.*weights)/sum(dist);
            else
                sfr(row,col,slice)=sum(dist(sp))/sum(dist);
            end
            
            sgm(row,col,slice)=exp(dot(dist(sp),log(T2_times(sp)))./sum(dist(sp)));
            mfr(row,col,slice)=sum(dist(mp))/sum(dist);
            mgm(row,col,slice)=exp(dot(dist(mp),log(T2_times(mp)))./sum(dist(mp)));
        end
    end
    %time_left=(toc(tstart2)/row)*(nrows-row)/60;
    %message=sprintf('%4.0f%% Complete\nTime Remaining: %2.0f minutes, %2.0f seconds',100*row/nrows,floor(time_left),(time_left-floor(time_left))*60);
    %waitbar(row/nrows,h,message)
end

maps.sfr=sfr;
maps.sgm=sgm;
maps.mfr=mfr;
maps.mgm=mgm;

% close(h);

% time=sprintf('%2.0f minutes, %2.0f seconds',floor(toc(tstart)/60),(toc(tstart)/60-floor(toc(tstart)/60))*60);
% name=inputname(1);
% fprintf('%s completed in %s\n',name,time);

toc(tstart);

end
