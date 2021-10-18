function clouds = stltranslateAll( clouds,r0 )
% translates the x,y,z coordinates of clouds by r0 (3D-vector)

%% translate point clouds
f=fieldnames(clouds);
notTrans={'isOrig','isPhys','geo','basis'};
for n = 1:numel(f)
    if ~any(strcmp(f{n},notTrans)); clouds.(f{n}){2}=trans(r0,clouds.(f{n}){2}); end;
end
clouds.isOrig=(norm(sphereFit(clouds.physis{2}))<1e-6);

%% translate geometric figures
f=fieldnames(clouds.geo);
for n = 1:numel(f)
    if strcmpi(f{n},'physisPlane'); ind=2;
    elseif strcmpi(f{n},'southWick'); continue;
    else ind=1; % special case
    end
    
    for i=1:3
        clouds.geo.(f{n}){ind}(i)=clouds.geo.(f{n}){ind}(i)-r0(i);
    end
end

end

