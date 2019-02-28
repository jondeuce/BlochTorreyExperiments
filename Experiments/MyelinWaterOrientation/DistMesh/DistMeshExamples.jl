module DistMeshExamples

using DistMesh
using Tensors
using GeometryUtils
using BenchmarkTools

function runbenchmarks()
    T = Float64
    V = Vec{2,T}
    h0 = T(0.05)
    bbox = T[-1 -1; 1 1]
    pfix = V[V((1,0)), V((0,1)), V((-1,0)), V((0,-1))]
    fd(x) = ddiff(norm(x) - 1, norm(x-V((-0.2,-0.2))) - 0.4)
    fh(x) = 1 + 2*norm(x)

    distmesh2d_(;kwargs...) = distmesh2d(fd, fh, h0, bbox, pfix; MAXSTALLITERS = 500, DETERMINISTIC = true, PLOT = false, kwargs...)
    display(@benchmark $distmesh2d_())
    distmesh2d_(;PLOTLAST = true)

    kmg2d_(;kwargs...) = kmg2d(fd, [], fh, h0, bbox, 1, 0, pfix; MAXITERS = 500, DETERMINISTIC = true, PLOT = false, kwargs...)
    display(@benchmark $kmg2d_())
    kmg2d_(;PLOTLAST = true)

    nothing
end

function runexamples()
    to_vec(P) = reinterpret(Vec{2,eltype(P)}, transpose(P)) |> vec |> copy

    # Example: (Uniform Mesh on Unit Circle)
    fd = p -> norm(p) - 1
    fh = huniform
    h0 = 0.2
    bbox = [-1.0 -1.0; 1.0 1.0]
    p, t = distmesh2d(fd, fh, h0, bbox; PLOTLAST = true)

    # Example: (Rectangle with circular hole, refined at circle boundary)
    fd = p -> ddiff(drectangle0(p,-1.0,1.0,-1.0,1.0), dcircle(p,0.0,0.0,0.5))
    fh = p -> 0.05 + 0.3 * dcircle(p,0.0,0.0,0.5)
    h0 = 0.05
    bbox = [-1.0 -1.0; 1.0 1.0]
    pfix = to_vec([-1.0 -1.0; -1.0 1.0; .01 -1.0; 1.0 1.0])
    p, t = distmesh2d(fd, fh, h0, bbox, pfix; PLOTLAST = true);

    nothing
end

function runblocktorreygrid()
    # btparams = BlochTorreyParameters{Float64}(AxonPDensity = 0.75, g_ratio = 0.78)
    # outercircles = packcircles(btparams;
    #     N = 10, # number of circles
    #     ϵ = 0.1 * btparams.R_mu, # overlap occurs when distance between circle edges is ≤ ϵ
    #     maxiter = 10 # maximum attempts for sampling radii + greedy packing + energy packing
    # )
    # innercircles = scale_shape.(outercircles, btparams.g_ratio)
    
    # Example circles from running the above circle packing:
    T = Float64
    V = Vec{2,T}
    centres = V[V((0.0473759, 0.0717498)), V((1.20823, 0.146365)), V((0.468667, 0.647647)), V((-0.536947, 0.947319)), V((0.829234, 1.11387)), V((1.76985, 1.25328)), V((0.237131, 1.01989)), V((0.988679, 1.93488)), V((0.294656, 1.566)), V((-0.328417, 1.82669))]
    outerradii = T[0.4704943963977774, 0.6329476737763889, 0.18858116384607837, 0.52365339677736110, 0.3478092493555470, 0.5457979689484037, 0.19859144852274970, 0.43261692334227597, 0.29802246626682216, 0.32337401293247753]
    innerradii = T[0.3669856291902664, 0.4936991855455834, 0.14709330779994115, 0.40844964948634166, 0.2712912144973267, 0.4257224157797549, 0.15490132984774477, 0.33744120020697527, 0.23245752368812128, 0.25223173008733246]
    rectlower, rectupper = V((-0.382468, 0.172455)), V((1.37816, 1.93308))
    
    rect = Rectangle{2,T}(rectlower, rectupper)
    outercircles = Circle{2,T}.(centres, outerradii)
    innercircles = Circle{2,T}.(centres, innerradii)
    mincircdist = minimum_signed_edge_distance(outercircles)
    
    p = plot(rect);
    for (ci,co) in zip(outercircles,innercircles)
        plot!(p,ci)
        plot!(p,co)
    end
    display(p)
    
    function dgrid(x::Vec{2,T}) where {T}
        douters, dinners = T(dunion()), T(dunion())
        @inbounds for i in eachindex(centres)
            c, ri, ro = centres[i], innerradii[i], outerradii[i]
            douters = dunion(douters, dcircle(x, c, ro))
            dinners = dunion(dinners, dcircle(x, c, ri))
        end
        drect = drectangle0(x, rectlower, rectupper)
        dext = ddiff(drect, douters)
        dtori = ddiff(douters, dinners)
        d = dintersect(drect, dunion(dext, dtori, dinners))
        return d
    end

    # Compute medial axis points
    h0 = mincircdist/2
    bbox = [xmin(rect) ymin(rect); xmax(rect) ymax(rect)]
    pfix = V[corners(rect)...]
    fd = dgrid
    
    # fh = hgeom(fd, h0, bbox; alpha = 0.5)
    # # pmedial, _ = DistMesh.medial_axis_search(fd, h0, bbox)
    dmax = maximum(x -> abs(fd(x)), DistMesh.cartesian_grid_generator(bbox, h0/4))
    fh = x -> 0.5 + abs(fd(x))/dmax

    # p, t = kmg2d(fd, [], fh, h0, bbox, 1, 0, pfix;
    #     MAXITERS = 500,
    #     VERBOSE = true,
    #     DETERMINISTIC = true,
    #     PLOT = false,
    #     PLOTLAST = true)

    return fd, fh, h0, bbox, pfix#, p, t#, pmedial
end

end # module DistMeshExamples

nothing
