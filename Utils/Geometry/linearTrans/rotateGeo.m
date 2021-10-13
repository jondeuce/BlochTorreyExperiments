function clouds = rotateGeo( clouds, opts )
% fields for structure opts:
%
%   -opts.parts: cell array of relevant parts being rotated
%       e.g. {'femur';'femurhead'}
%   -opts.angles: cell array of double vectors of rotation angles
%       e.g. {[30;20;45];[10]}
%   -opts.axes: cell array of cell vectors of rotation axes
%       e.g. {{'y';'x';'z'};{'x'}}

opts.parts=opts.parts(:);
opts.angles=opts.angles(:);
opts.axes=opts.axes(:);
if ~all(size(opts.parts,1)==[size(opts.angles,1) size(opts.axes,1)])
    error(['ERROR (rotateGeo): each part needs '...
        'a matching angle vector and axes vector']);
else
    for i=1:size(opts.parts,1)
        opts.angles{i}=opts.angles{i}(:);
        opts.axes{i}=opts.axes{i}(:);
        if length(opts.angles{i})~=length(opts.axes{i})
            error(['ERROR (rotateGeo): each angle must have a '...
                'specified axis to rotate about']);
        end
    end
end

% femur: physis sphere, physis ellipsoid, neck cylinder
% femurhead: capital sphere, capital vector, physis plane,

for n=1:size(opts.parts,1)
    if size(opts.angles{n},1)>1
        m=length(opts.angles{n}); args=cell(2*(m-1),1);
        for i=2:m
            args{2*i-3}=opts.angles{n}(i);
            args{2*i-2}=opts.axes{n}{i};
        end
    else args={};
    end
    
    if strcmp(opts.parts{n},'femur')
        clouds.geo.physisSphere{1}=(rotateCloud(clouds.geo.physisSphere{1}',...
            opts.angles{n}(1),opts.axes{n}{1},args{:}))';
        clouds.geo.physisEllipsoid{1}=(rotateCloud(clouds.geo.physisEllipsoid{1}',...
            opts.angles{n}(1),opts.axes{n}{1},args{:}))';
        for i=1:2
            clouds.geo.neckCylinder{i}=(rotateCloud(clouds.geo.neckCylinder{i}',...
                opts.angles{n}(1),opts.axes{n}{1},args{:}))';
        end
    elseif strcmp(opts.parts{n},'femurhead')
        clouds.geo.capitalSphere{1}=(rotateCloud(clouds.geo.capitalSphere{1}',...
            opts.angles{n}(1),opts.axes{n}{1},args{:}))';
        clouds.geo.capitalVector{2}=(rotateCloud(clouds.geo.capitalVector{2}',...
            opts.angles{n}(1),opts.axes{n}{1},args{:}))';
        for i=1:3
            clouds.geo.physisPlane{i}=(rotateCloud(clouds.geo.physisPlane{i}',...
                opts.angles{n}(1),opts.axes{n}{1},args{:}))';
        end
    else
        error(['ERROR (rotateGeo): the only supported '...
            'parts are ''femur'' and ''femurhead''']);
    end
end

end