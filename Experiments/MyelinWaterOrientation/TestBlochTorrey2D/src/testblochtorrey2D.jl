function testblochtorrey2D(;
        Npts = 100, # Npts per dim for finite differences
        Resolution = 1.0, # Resolution factor for finite elem. Smaller value reduces triangle size
        D = 1.0 # isotropic diffusion constant
    )
    T = Float64

    p = BlochTorreyParameters{T}(
        D_Tissue = D, D_Axon = D, D_Sheath = D, D_Water = D, D_Blood = D,
        K_perm = 0.0
    )

    sols, FE = testbtfinelem2D(p; Resolution = Resolution)
    U, FD = testbtfindiff2D(p; Npts = Npts)

    Xq = [x for x in FD.pts, y in FD.pts]
    Yq = [y for x in FD.pts, y in FD.pts]

    U_FE = zeros(Complex{T}, size(U))
    shifts = [0; cumsum(ndofs.(FE.myelinsubdomains[1:end-1]))] .+ 1
    for (i,m) in enumerate(FE.myelinsubdomains)
        nodelists = getnodes(getgrid(getdomain(m)))
        coordlist = reinterpret(T, getcoordinates.(nodelists))
        X = coordlist[1:2:end]
        Y = coordlist[2:2:end]
        # U_end = sols[1].u[end]
        # U_sub = i == length(shifts) ? U_end[shifts[i]:end] : U_end[shifts[i]:shifts[i+1]-1]
        U_sub = sols[i].u[end]
        V = reinterpret(Complex{T}, U_sub) |> copy

        spl(Xq::AbstractArray,Yq::AbstractArray) = mxcall(:griddata,1,X,Y,V,Xq,Yq)
        spl(Xq::Number,Yq::Number) = spl([Xq],[Yq])[1]

        @show getregion(m)
        b = if getregion(m) isa AxonRegion
            FD.m_int
        elseif getregion(m) isa TissueRegion
            FD.m_ext
        else # MyelinRegion
            .!(FD.m_int .| FD.m_ext)
        end
        U_FE[b] .= spl(Xq, Yq)[b]
    end

    # pick(x,y) = ifelse(isnan(x), y, x)
    # U_FE = reduce((X,Y) -> pick.(X,Y), U_FEs)

    return U, U_FE, sols, FE, FD
end

# using Plots; pyplot()
# Plots.scalefontsizes(2)
# heatmap(abs.(U.-U_FE))
