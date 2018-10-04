function [moreechoimages,parms]=read_parec_moreechoes_AW(parfile)
%clear all; close all;
% reads Philips PAR/REC files 
%
% Inputs:
%           If no inputs, via UI, select name of PAR file
%           assumes the REC file in the same folder as PAR file
%           With inputs:
%                       filename = filename (with path) of PAR file
%
% Output:
% Can process both from V3 upto V4.2 types of PAR files
%
% Steps:
% 1: Read PAR/REC data in matrix
%==========================================================================
% Author        Amol Pednekar
%               MR Clinical Science
%               Philips medical systems
%               3/22/2007

% Jing Zhang  This program was modified to import the more echoes images. 
%==========================================================================
% if nargin < 1
%     [fname,pname] = uigetfile('*.PAR','Select *.PAR file');
%     parfile=[pname fname];
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
parfile = char(parfile);

tic;
[parms,ver] = read_parfile(parfile);
[image_data,dims,nidx] = read_recfile([parfile(1:(end-4)) '.REC'],parms,ver);
%%%------
% %Jing add this to save the image_data before it disappear
image=permute(image_data,[1,2,3,5,4]); % exchange the number of echoes and number of cardiac phases.
moreechoimages=image(:,:,:,:,1);   % the 5th dimension is the number of echoes. The 4th dimension only used 1 echoes. 
%save importedimage moreechoeimages
%%%----------------------
disp(sprintf('Read %d images of size [%d x %d] \n from %s',prod(dims(3:end)),dims(1:2),parfile(1:(end-4))));
disp(sprintf('# slices: %d, # echoes: %d, # phases: %d, # types: %d, # indices: %d, # dynamics: %d',dims(3:end)));

% figure, hold on;
% imshow(image_data(:,:,1,1,1,1,1,1,1),[]);   % Jing commented these two
% lines because it is slow to show. 

%==========================================================================

function [parms,ver] = read_parfile(file_name)

%==========================================================================

% read all the lines of the PAR file
nlines  = 0;
fid = fopen(file_name,'r');
if (fid < 1), error(['.PAR file ', file_name, ' not found.']); end;
while ~feof(fid)
    curline = fgetl(fid);
    if ~isempty(curline)
        nlines = nlines + 1;
        lines(nlines) = cellstr(curline); % allows variable line size
    end
end
fclose(fid);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% identify the header lines
NG = 0; nimg  = 0;
ver_key = '# CLINICAL TRYOUT             Research image export tool     V4.2';
for L = 1:size(lines,2)
    curline = char(lines(L)); firstc=curline(1);
    if (length(curline)>18 & curline(1:17) == '# CLINICAL TRYOUT')
        ver = curline(strfind(curline,'V')+1:length(curline));
    elseif (size(curline,2) > 0 & firstc == '.')
        NG = NG + 1;
        geninfo(NG) = lines(L);
    elseif (size(curline,2) > 0 & firstc ~= '.' & firstc ~= '#' & firstc ~= '*')
        nimg = nimg + 1;
        parms.tags(nimg,:) = str2num(curline);
    end
end
if (nimg < 1), error('Missing scan information in .PAR file'); end;
if (NG < 1), error('.PAR file has invalid format'); end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure out if V3 or V4 PAR files
test_key = '.    Image pixel size [8 or 16 bits]    :';
if (strmatch('4.2',ver,'exact'))
    template=get_template_v42;
    disp(' ****** Version 4.2 *****');
elseif (strmatch('4',ver,'exact'))
    template=get_template_v4;
    disp(' ****** Version 4 *****');
else
    template=get_template_v3;
    disp(' ****** Version 3 *****');
end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% parse the header information
for S=1:size(template,1)
    line_key = char(template(S,1));
    value_type = char(template(S,2));
    field_name = char(template(S,3));
    L = strmatch(line_key,geninfo);

    if ~isempty(L)
        curline = char(geninfo(L));
        value_start = 1 + strfind(curline,':');
        value_end = size(curline,2);
    else
        value_type = ':-( VALUE NOT FOUND )-:';
    end

    switch (value_type)
        case { 'float scalar' 'int   scalar' 'float vector' 'int   vector'}
            parms.(field_name) = str2num(curline(value_start:value_end));
        case { 'char  scalar' 'char  vector' }
            parms.(field_name) = deblank(strjust(curline(value_start:value_end),'left'));
        otherwise
            parms.(field_name) = '';
    end

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


return;

%==========================================================================

function [image_data,dims,nidx] = read_recfile(recfile_name,parms,ver);

%==========================================================================
types_list = unique(parms.tags(:,5)'); % to handle multiple types
idx_list  = unique(parms.tags(:,6)'); % to handle multiple indices

scan_tag_size = size(parms.tags);
nimg = scan_tag_size(1);
nslice = parms.max_slices;
nphase = parms.max_card_phases;
necho = parms.max_echoes;
ndyn = parms.max_dynamics;
ntype = size(types_list,2);
nidx  = size(idx_list,2);
if (strmatch('4.2',ver,'exact'))
    nLabel = parms.No_of_label_types;
else
    nLabel = 1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ( isfield(parms,'recon_resolution') )
    nline = parms.recon_resolution(1);
    stride = parms.recon_resolution(2);
else
    nline = parms.tags(1,10);
    stride = parms.tags(1,11);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
switch(ver)
    case {'3'}, pixel_bits = parms.pixel_bits;
    case {'4'}, pixel_bits = parms.tags(1,8);  % assume same for all imgs
    case {'4.2'}, pixel_bits = parms.tags(1,8);  % assume same for all imgs
end

switch (pixel_bits)
    case { 8 }, read_type = 'int8';
    case { 16 }, read_type = 'short';
    otherwise, read_type = 'uchar';
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% read the REC file
fid = fopen(recfile_name,'r','l');
[binary_1D,read_size] = fread(fid,inf,read_type);
fclose(fid);

if (read_size ~= nimg*nline*stride)
    disp(sprintf('Expected %d int.  Found %d int',nimg*nline*stride,read_size));
    if (read_size > nimg*nline*stride)
        error('.REC file has more data than expected from .PAR file')
    else
        error('.REC file has less data than expected from .PAR file')
    end
else
    disp(sprintf('.REC file read sucessfully'));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% generate the final matrix of images
if (strmatch('4.2',ver,'exact'))
    dims = [stride nline nslice necho nphase ntype nidx ndyn nLabel];
else
    dims = [stride nline nslice necho nphase ntype nidx ndyn];
end
image_data=zeros(dims);
for I  = 1:nimg
    slice = parms.tags(I,1);
    echo = parms.tags(I,2);
    dyn = parms.tags(I,3);
    phase = parms.tags(I,4);
    type = parms.tags(I,5);
    type_num = find(types_list == type);
    idx = parms.tags(I,6);
    idx_num = find(idx_list == idx);
    rec = parms.tags(I,7);
    start_image = 1+(rec*nline*stride);
    end_image = start_image + stride*nline - 1;
    img = reshape(binary_1D(start_image:end_image),stride,nline);
    % rescale data to produce FP information (not SV, not DV)
    img = permute(rescale_rec(img,parms.tags(I,:),ver), [2 1]);
    
    
% Jing start to add and get T2 decay the 64 echoes with 40 slices data.
   image64echoes40slices(:,:,slice,echo,phase)=img;
   
% Jing---------------------

% Jing------comment on the following line to import 64 echoes with 40 slices data.
%     if (strmatch('4.2',ver,'exact'))
%         label = parms.tags(I,49);
%         image_data(:,:,slice,echo,phase,type_num,idx_num,dyn,label) = img;
%     else
%         label = 1;
%         image_data(:,:,slice,echo,phase,type_num,idx_num,dyn) = img;
%     end
% Jing----------------------------------------------------------------------------  

    
    
end


image_data=image64echoes40slices; % Jing start to add and get T2 decay the 64 echoes with 40 slices data.

return;

%==========================================================================

function img = rescale_rec(img,tag,ver)

% transforms SV data in REC files to FP data
switch( ver )
    case { '3' }, ri_i = 8; rs_i = 9; ss_i = 10;
    case { '4' }, ri_i = 12; rs_i = 13; ss_i = 14;
    case { '4.2' }, ri_i = 12; rs_i = 13; ss_i = 14;
end;
RI = tag(ri_i);  % 'inter' --> 'RI'
RS = tag(rs_i);  % 'slope' --> 'RS'
SS = tag(ss_i);  % new var 'SS'
% img = (RS*img + RI)./(RS*SS);  

 img=img; % Jing-I took the scaling out for the T2 data because it is not needed for the GRASE sequence--one sequence, all scaling factors are same. 

return;

%==========================================================================
function [template] = get_template_v3;  % header information for V3 PAR files

template = { ...
    '.    Patient name                       :'    'char  scalar'    'patient';    ...
    '.    Examination name                   :'    'char  scalar'    'exam_name';   ...
    '.    Protocol name                      :'    'char  vector'    'protocol';   ...
    '.    Examination date/time              :'    'char  vector'    'exam_date';  ...
    '.    Acquisition nr                     :'    'int   scalar'    'acq_nr';    ...
    '.    Reconstruction nr                  :'    'int   scalar'    'recon_nr';  ...
    '.    Scan Duration [sec]                :'    'float scalar'    'scan_dur';        ...
    '.    Max. number of cardiac phases      :'    'int   scalar'    'max_card_phases'; ...
    '.    Max. number of echoes              :'    'int   scalar'    'max_echoes'; ...
    '.    Max. number of slices/locations    :'    'int   scalar'    'max_slices'; ...
    '.    Max. number of dynamics            :'    'int   scalar'    'max_dynamics'; ...
    '.    Max. number of mixes               :'    'int   scalar'    'max_mixes'; ...
    '.    Image pixel size [8 or 16 bits]    :'    'int   scalar'    'pixel_bits'; ...
    '.    Technique                          :'    'char  scalar'    'technique'; ...
    '.    Scan mode                          :'    'char  scalar'    'scan_mode'; ...
    '.    Scan resolution  (x, y)            :'    'int   vector'    'scan_resolution'; ...
    '.    Scan percentage                    :'    'int   scalar'    'scan_percentage'; ...
    '.    Recon resolution (x, y)            :'    'int   vector'    'recon_resolution'; ...
    '.    Number of averages                 :'    'int   scalar'    'num_averages'; ...
    '.    Repetition time [msec]             :'    'float scalar'    'repetition_time'; ...
    '.    FOV (ap,fh,rl) [mm]                :'    'float vector'    'fov'; ...
    '.    Slice thickness [mm]               :'    'float scalar'    'slice_thickness'; ...
    '.    Slice gap [mm]                     :'    'float scalar'    'slice_gap'; ...
    '.    Water Fat shift [pixels]           :'    'float scalar'    'water_fat_shift'; ...
    '.    Angulation midslice(ap,fh,rl)[degr]:'    'float vector'    'angulation'; ...
    '.    Off Centre midslice(ap,fh,rl) [mm] :'    'float vector'    'offcenter'; ...
    '.    Flow compensation <0=no 1=yes> ?   :'    'int   scalar'    'flowcomp'; ...
    '.    Presaturation     <0=no 1=yes> ?   :'    'int   scalar'    'presaturation';...
    '.    Cardiac frequency                  :'    'int   scalar'    'card_frequency'; ...
    '.    Min. RR interval                   :'    'int   scalar'    'min_rr_interval'; ...
    '.    Max. RR interval                   :'    'int   scalar'    'max_rr_interval'; ...
    '.    Phase encoding velocity [cm/sec]   :'    'float vector'    'venc'; ...
    '.    MTC               <0=no 1=yes> ?   :'    'int   scalar'    'mtc'; ...
    '.    SPIR              <0=no 1=yes> ?   :'    'int   scalar'    'spir'; ...
    '.    EPI factor        <0,1=no EPI>     :'    'int   scalar'    'epi_factor'; ...
    '.    TURBO factor      <0=no turbo>     :'    'int   scalar'    'turbo_factor'; ...
    '.    Dynamic scan      <0=no 1=yes> ?   :'    'int   scalar'    'dynamic_scan'; ...
    '.    Diffusion         <0=no 1=yes> ?   :'    'int   scalar'    'diffusion'; ...
    '.    Diffusion echo time [msec]         :'    'float scalar'    'diffusion_echo_time'; ...
    '.    Inversion delay [msec]             :'    'float scalar'    'inversion_delay'; ...
    };

return;

%==========================================================================
function [template] = get_template_v4;    % header information for V4 PAR files

template = { ...
    '.    Patient name                       :' 'char  scalar' 'patient';    ...
    '.    Examination name                   :' 'char  vector' 'exam_name';   ...
    '.    Protocol name                      :' 'char  vector' 'protocol';   ...
    '.    Examination date/time              :' 'char  vector' 'exam_date';  ...
    '.    Series Type                        :' 'char  vector' 'series_type';  ...
    '.    Acquisition nr                     :' 'int   scalar' 'acq_nr';    ...
    '.    Reconstruction nr                  :' 'int   scalar' 'recon_nr';  ...
    '.    Scan Duration [sec]                :' 'float scalar' 'scan_dur';        ...
    '.    Max. number of cardiac phases      :' 'int   scalar' 'max_card_phases'; ...
    '.    Max. number of echoes              :' 'int   scalar' 'max_echoes'; ...
    '.    Max. number of slices/locations    :' 'int   scalar' 'max_slices'; ...
    '.    Max. number of dynamics            :' 'int   scalar' 'max_dynamics'; ...
    '.    Max. number of mixes               :' 'int   scalar' 'max_mixes'; ...
    '.    Patient position                   :' 'char  vector' 'patient_position'; ...
    '.    Preparation direction              :' 'char  vector' 'preparation_dir'; ...
    '.    Technique                          :' 'char  scalar' 'technique'; ...
    '.    Scan resolution  (x, y)            :' 'int   vector' 'scan_resolution'; ...
    '.    Scan mode                          :' 'char  scalar' 'scan_mode'; ...
    '.    Repetition time [ms]               :' 'float scalar' 'repetition_time'; ...
    '.    FOV (ap,fh,rl) [mm]                :' 'float vector' 'fov'; ...
    '.    Water Fat shift [pixels]           :' 'float scalar' 'water_fat_shift'; ...
    '.    Angulation midslice(ap,fh,rl)[degr]:' 'float vector' 'angulation'; ...
    '.    Off Centre midslice(ap,fh,rl) [mm] :' 'float vector' 'offcenter'; ...
    '.    Flow compensation <0=no 1=yes> ?   :' 'int   scalar' 'flowcomp'; ...
    '.    Presaturation     <0=no 1=yes> ?   :' 'int   scalar' 'presaturation';...
    '.    Phase encoding velocity [cm/sec]   :' 'float vector' 'venc'; ...
    '.    MTC               <0=no 1=yes> ?   :' 'int   scalar' 'mtc'; ...
    '.    SPIR              <0=no 1=yes> ?   :' 'int   scalar' 'spir'; ...
    '.    EPI factor        <0,1=no EPI>     :' 'int   scalar' 'epi_factor'; ...
    '.    Dynamic scan      <0=no 1=yes> ?   :' 'int   scalar' 'dynamic_scan'; ...
    '.    Diffusion         <0=no 1=yes> ?   :' 'int   scalar' 'diffusion'; ...
    '.    Diffusion echo time [msec]         :' 'float scalar' 'diffusion_echo_time'; ...
    };

return;
%==========================================================================
function [template] = get_template_v42;    % header information for V4.2 PAR files

template = { ...
    '.    Patient name                       :' 'char  scalar' 'patient';    ...
    '.    Examination name                   :' 'char  vector' 'exam_name';   ...
    '.    Protocol name                      :' 'char  vector' 'protocol';   ...
    '.    Examination date/time              :' 'char  vector' 'exam_date';  ...
    '.    Series Type                        :' 'char  vector' 'series_type';  ...
    '.    Acquisition nr                     :' 'int   scalar' 'acq_nr';    ...
    '.    Reconstruction nr                  :' 'int   scalar' 'recon_nr';  ...
    '.    Scan Duration [sec]                :' 'float scalar' 'scan_dur';        ...
    '.    Max. number of cardiac phases      :' 'int   scalar' 'max_card_phases'; ...
    '.    Max. number of echoes              :' 'int   scalar' 'max_echoes'; ...
    '.    Max. number of slices/locations    :' 'int   scalar' 'max_slices'; ...
    '.    Max. number of dynamics            :' 'int   scalar' 'max_dynamics'; ...
    '.    Max. number of mixes               :' 'int   scalar' 'max_mixes'; ...
    '.    Patient position                   :' 'char  vector' 'patient_position'; ...
    '.    Preparation direction              :' 'char  vector' 'preparation_dir'; ...
    '.    Technique                          :' 'char  scalar' 'technique'; ...
    '.    Scan resolution  (x, y)            :' 'int   vector' 'scan_resolution'; ...
    '.    Scan mode                          :' 'char  scalar' 'scan_mode'; ...
    '.    Repetition time [ms]               :' 'float scalar' 'repetition_time'; ...
    '.    FOV (ap,fh,rl) [mm]                :' 'float vector' 'fov'; ...
    '.    Water Fat shift [pixels]           :' 'float scalar' 'water_fat_shift'; ...
    '.    Angulation midslice(ap,fh,rl)[degr]:' 'float vector' 'angulation'; ...
    '.    Off Centre midslice(ap,fh,rl) [mm] :' 'float vector' 'offcenter'; ...
    '.    Flow compensation <0=no 1=yes> ?   :' 'int   scalar' 'flowcomp'; ...
    '.    Presaturation     <0=no 1=yes> ?   :' 'int   scalar' 'presaturation';...
    '.    Phase encoding velocity [cm/sec]   :' 'float vector' 'venc'; ...
    '.    MTC               <0=no 1=yes> ?   :' 'int   scalar' 'mtc'; ...
    '.    SPIR              <0=no 1=yes> ?   :' 'int   scalar' 'spir'; ...
    '.    EPI factor        <0,1=no EPI>     :' 'int   scalar' 'epi_factor'; ...
    '.    Dynamic scan      <0=no 1=yes> ?   :' 'int   scalar' 'dynamic_scan'; ...
    '.    Diffusion         <0=no 1=yes> ?   :' 'int   scalar' 'diffusion'; ...
    '.    Diffusion echo time [msec]         :' 'float scalar' 'diffusion_echo_time'; ...
    '.    Max. number of diffusion values    :' 'int   scalar' 'Max_diffusion_values'; ...
    '.    Max. number of gradient orients    :' 'int   scalar' 'Max_gradient_orients'; ...
    '.    Number of label types   <0=no ASL> :' 'int   scalar' 'No_of_label_types'; ...
    };

return;

%==========================================================================
function [R] = imscale(I,lowout,highout)

[a, b] = size(size(I));
I = double(I);
lowin = min(min(min(I)));
highin = max(max(max(I)));
R = lowout + (highout - lowout)*(I -lowin)/(highin-lowin);
