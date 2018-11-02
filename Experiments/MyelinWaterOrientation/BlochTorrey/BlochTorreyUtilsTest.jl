module BlochTorreyUtilsTest

using BlochTorreyUtils
using GeometryUtils
using CirclePackingUtils
using GreedyCirclePacking
using MeshUtils
using Tensors
using JuAFEM
using MATLAB

using BlochTorreyUtils: omega_tissue, omega_myelin, omega_axon, omega, omegamap, dcoeff, rdecay
using BlochTorreyUtils: AxonRegion, MyelinRegion, TissueRegion

using Test
using BenchmarkTools
using Parameters

const BTParams{T} = BlochTorreyParameters{T}

# ---------------------------------------------------------------------------- #
# Single axon tests
# ---------------------------------------------------------------------------- #

function singleaxon(
        btparams::BTParams{T} = BTParams{Float64}(
            theta = pi/2,
            g_ratio = 0.8
        )
    ) where {T}

    Dim = 2
    Ncircles = 1
    V = Vec{2,T}

    @unpack R_mu, g_ratio = btparams

    rs = [R_mu]
    bdry = Rectangle(-2R_mu*ones(V), 2R_mu*ones(V))
    c_out, c_in = Circle(origin(bdry), R_mu), Circle(origin(bdry), g_ratio * R_mu)

    h0 = (0.5 + 0.5*rand())*(1-g_ratio)*R_mu # fraction of size of minimum torus width
    h_min = h0 # minimum edge length
    h_max = 5*h0 # maximum edge length
    h_range = 10*h0 # distance over which h increases from h_min to h_max
    h_rate = 0.6 # rate of increase of h from circle boundaries (power law; smaller = faster radial increase)

    @time exteriorgrids, torigrids, interiorgrids, parentcircleindices = disjoint_rect_mesh_with_tori(
        bdry, [c_in], [c_out], h_min, h_max, h_range, h_rate;
        maxstalliters = 1000, plotgrids = false, exterior_tiling = (1, 1)
    )

    myelindomains = createmyelindomains(
        exteriorgrids, torigrids, interiorgrids, [c_out], [c_in];
        quadorder = 3, funcinterporder = 1
    )

    return myelindomains, c_in, c_out, bdry
end

function singleaxontests(
        btparams::BTParams{T} = BTParams{Float64}(
            theta = pi/2,
            g_ratio = 0.8
        ),
        domainsetup = singleaxon(btparams);
        RTOL = T(1e-14),
        PLOTOMEGA = false
    ) where {T}

    @testset "Single Axon" begin
        myelindomains, c_in, c_out, bdry = domainsetup

        if PLOTOMEGA
            mxcall(:figure, 0)
            omegavalues = omegamap(MyelinProblem(btparams), myelindomains)
            for (m, w) in zip(myelindomains, omegavalues)
                simpplot(getgrid(m); newfigure = false, hold = true, axis = mxaxis(bdry), facecol = w);
            end
        end

        for testtype in (:isotropic, :anisotropic, :total)
            chiA = testtype == :isotropic ? zero(T) : btparams.ChiA
            chiI = testtype == :anisotropic ? zero(T) : btparams.ChiI
            params = BlochTorreyParameters(btparams; ChiI = chiI, ChiA = chiA)

            @unpack B0, gamma, theta, ChiI, ChiA, E, g_ratio = params
            ω₀ = gamma * B0
            ωI, ωA, ωE = ω₀*ChiI, ω₀*ChiA, ω₀*E
            s², c² = sin(theta)^2, cos(theta)^2

            prob = MyelinProblem(params)
            omegavalues = omegamap(prob, myelindomains)

            for (m, w) in zip(myelindomains, omegavalues)
                region = getregion(m)

                xs = getcoordinates.(getnodes(getgrid(m)))
                xs_out = filter(x -> abs(norm(x) - params.R_mu) ≤ 100*eps(T), xs)
                xs_in = filter(x -> abs(norm(x) - g_ratio * params.R_mu) ≤ 100*eps(T), xs)

                cos2ϕ(x::Vec{2}) = (ϕ = atan(x[2],x[1]); cos(2ϕ))

                c2ϕ_max_out = isempty(xs_out) ? one(T) : cos2ϕ(xs_out[findmin([abs(x[2]) for x in xs_out])[2]])
                c2ϕ_min_out = isempty(xs_out) ? one(T) : cos2ϕ(xs_out[findmin([abs(x[1]) for x in xs_out])[2]])
                c2ϕ_max_in  = isempty(xs_in)  ? one(T) : cos2ϕ(xs_in[findmin([abs(x[2]) for x in xs_in])[2]])
                c2ϕ_min_in  = isempty(xs_in)  ? one(T) : cos2ϕ(xs_in[findmin([abs(x[1]) for x in xs_in])[2]])

                if isa(region, AxonRegion) && testtype == :isotropic
                    @test maximum(w) == zero(T)
                    @test minimum(w) == zero(T)
                elseif isa(region, AxonRegion) && testtype == :anisotropic
                    @test maximum(w) ≈ ωA * s² * -3*log(g_ratio)/4 rtol = 5*eps(T) # should be exact
                    @test minimum(w) ≈ ωA * s² * -3*log(g_ratio)/4 rtol = 5*eps(T) # should be exact
                elseif isa(region, AxonRegion) && testtype == :total
                    @test maximum(w) ≈ ωA * s² * -3*log(g_ratio)/4 rtol = 5*eps(T) # should be exact
                    @test minimum(w) ≈ ωA * s² * -3*log(g_ratio)/4 rtol = 5*eps(T) # should be exact
                elseif isa(region, MyelinRegion) && testtype == :isotropic
                    @test maximum(w) ≈ ωE + ωI * (c² - 1/3 - c2ϕ_min_in * s²)/2 rtol = RTOL
                    @test minimum(w) ≈ ωE + ωI * (c² - 1/3 - c2ϕ_max_in * s²)/2 rtol = RTOL
                elseif isa(region, MyelinRegion) && testtype == :anisotropic
                    @test maximum(w) ≈ ωE + ωA * s² * (-5/12 - c2ϕ_min_in * 2/8 - 3*log(g_ratio)/4) - ωA * c²/6 rtol = RTOL
                    @test minimum(w) ≈ ωE + ωA * s² * (-5/12 - c2ϕ_max_out * 1/8 * (1 + g_ratio^2)) - ωA * c²/6 rtol = RTOL
                elseif isa(region, MyelinRegion) && testtype == :total
                    @test maximum(w) ≈ ωE + ωI * (c² - 1/3 - c2ϕ_min_in * s²)/2 + ωA * s² * (-5/12 - c2ϕ_min_in * 2/8 - 3*log(g_ratio)/4) - ωA * c²/6 rtol = RTOL
                    @test minimum(w) ≈ ωE + ωI * (c² - 1/3 - c2ϕ_max_out * s² * g_ratio^2)/2 + ωA * s² * (-5/12 - c2ϕ_max_out * 1/8 * (1 + g_ratio^2)) - ωA * c²/6 rtol = RTOL
                elseif isa(region, TissueRegion) && testtype == :isotropic
                    @test maximum(w) ≈ ωI * s² * c2ϕ_max_out * (1 - g_ratio^2)/2 rtol = RTOL
                    @test minimum(w) ≈ ωI * s² * c2ϕ_min_out * (1 - g_ratio^2)/2 rtol = RTOL
                elseif isa(region, TissueRegion) && testtype == :anisotropic
                    @test maximum(w) ≈ ωA * s² * c2ϕ_max_out * (1 - g_ratio^2)/8 rtol = RTOL
                    @test minimum(w) ≈ ωA * s² * c2ϕ_min_out * (1 - g_ratio^2)/8 rtol = RTOL
                elseif isa(region, TissueRegion) && testtype == :total
                    @test maximum(w) ≈ ωI * s² * c2ϕ_max_out * (1 - g_ratio^2)/2 + ωA * s² * c2ϕ_max_out * (1 - g_ratio^2)/8 rtol = RTOL
                    @test minimum(w) ≈ ωI * s² * c2ϕ_min_out * (1 - g_ratio^2)/2 + ωA * s² * c2ϕ_min_out * (1 - g_ratio^2)/8 rtol = RTOL
                end
            end
        end

    end

    nothing
end


# ---------------------------------------------------------------------------- #
# Multiple axon tests
# ---------------------------------------------------------------------------- #

function multipleaxons(
        btparams::BTParams{T} = BTParams{Float64}(
            theta = pi/2,
            g_ratio = 0.8
        )
    ) where {T}

    Dim = 2
    Ncircles = 20
    V = Vec{Dim,T}

    rs = rand(radiidistribution(btparams), Ncircles)
    initial_circles = GreedyCirclePacking.pack(rs; iters = 100)
    outer_circles = scale_to_density(initial_circles, 0.6)
    inner_circles = scale_shape.(outer_circles, btparams.g_ratio)
    bdry, _ = opt_subdomain(outer_circles)

    h0 = (1-btparams.g_ratio)/2 * minimum(rs) # fraction of size of minimum torus width
    h_min = h0 # minimum edge length
    h_max = 5*h0 # maximum edge length
    h_range = 10*h0 # distance over which h increases from h_min to h_max
    h_rate = 0.6 # rate of increase of h from circle boundaries (power law; smaller = faster radial increase)

    @time exteriorgrids, torigrids, interiorgrids, parentcircleindices = disjoint_rect_mesh_with_tori(
        bdry, inner_circles, outer_circles, h_min, h_max, h_range, h_rate;
        maxstalliters = 1000, plotgrids = false, exterior_tiling = (2, 2)
    )

    myelindomains = createmyelindomains(
        exteriorgrids, torigrids, interiorgrids, outer_circles, inner_circles;
        quadorder = 3, funcinterporder = 1
    )

    return myelindomains, inner_circles, outer_circles, bdry
end

function multipleaxontests(
        btparams::BTParams{T} = BTParams{Float64}(
            theta = pi/2,
            g_ratio = 0.8
        ),
        domainsetup = multipleaxons(btparams);
        RTOL = T(1e-14),
        PLOTOMEGA = false
    ) where {T}

    @testset "Multiple Axons" begin
        myelindomains, inner_circles, outer_circles, bdry = domainsetup
        prob = MyelinProblem(btparams)

        if PLOTOMEGA
            mxcall(:figure, 0)
            omegavalues = omegamap(prob, myelindomains)
            for (m, w) in zip(myelindomains, omegavalues)
                simpplot(getgrid(m); newfigure = false, hold = true, axis = mxaxis(bdry), facecol = w);
            end
        end

        omegavalues = omegamap(prob, myelindomains)
        omegatest = map(myelindomains, omegavalues) do m, w
            outer_bdry_point_type = isa(getregion(m), MyelinRegion) ? :myelin : :tissue
            inner_bdry_point_type = isa(getregion(m), MyelinRegion) ? :myelin : :axon
            map(getnodes(getgrid(m))) do node
                x = getcoordinates(node)
                sum(zip(getoutercircles(m), getinnercircles(m))) do (c_out, c_in)
                    omega(x, prob, [c_out], [c_in], outer_bdry_point_type, inner_bdry_point_type)
                end
            end
        end

        @test omegavalues == omegatest
    end

end

end # module BlochTorreyUtilsTest

nothing
