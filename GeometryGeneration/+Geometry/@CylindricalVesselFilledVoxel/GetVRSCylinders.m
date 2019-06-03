function [p, r, vx, vy, vz] = GetVRSCylinders(G)

    [p, r, vx, vy, vz] = deal(G.p0, G.r0, G.vx0, G.vy0, G.vz0);
    
    if ~isempty(G.MediumVesselRadiusThresh)
        p = [p, G.p(:, G.MediumVessels)];
        r = [r, G.r(:, G.MediumVessels)];
        vx = [vx, G.vx(:, G.MediumVessels)];
        vy = [vy, G.vy(:, G.MediumVessels)];
        vz = [vz, G.vz(:, G.MediumVessels)];
    end
    r = G.VRSRelativeRad * r;
    
end