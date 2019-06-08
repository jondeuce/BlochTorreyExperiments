# using DistMesh
# using Tensors
# using GeometryUtils
using CirclePackingUtils
# using MeshUtils
using BlochTorreyUtils
using MWFUtils

using LinearAlgebra, Statistics, BenchmarkTools

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

function runblocktorreygrid(;alpha = 0.5, beta = 0.5, Ncircles = 10, PLOT = false)
    T = Float64
    V = Vec{2,T}
    btparams = BlochTorreyParameters{T}(AxonPDensity = 0.75, g_ratio = 0.78)
    
    outercircles, _ = packcircles(btparams;
        Ncircles = Ncircles, # number of circles
        distthresh = 0.1 * btparams.R_mu, # overlap occurs when distance between circle edges is ≤ ϵ
        maxattempts = 10 # maximum attempts for sampling radii + greedy packing + energy packing
    )
    innercircles = scale_shape.(outercircles, btparams.g_ratio)
    
    rect, _ = opt_subdomain(collect(Iterators.flatten(zip(outercircles, innercircles))); MODE = :corners) #DEBUG
    outercircles, rect, α_best = scale_to_density(outercircles, rect, btparams.AxonPDensity)
    innercircles = scale_shape.(outercircles, btparams.g_ratio)

    # # Example circles from running the above circle packing:
    # outercircles = [Circle{2,T}(V((0.0473759, 0.0717498)), 0.4704943963977774), Circle{2,T}(V((1.20823, 0.146365)), 0.6329476737763889), Circle{2,T}(V((0.468667, 0.647647)), 0.18858116384607837), Circle{2,T}(V((-0.536947, 0.947319)), 0.52365339677736110), Circle{2,T}(V((0.829234, 1.11387)), 0.3478092493555470), Circle{2,T}(V((1.76985, 1.25328)), 0.5457979689484037), Circle{2,T}(V((0.237131, 1.01989)), 0.19859144852274970), Circle{2,T}(V((0.988679, 1.93488)), 0.43261692334227597), Circle{2,T}(V((0.294656, 1.566)), 0.29802246626682216), Circle{2,T}(V((-0.328417, 1.82669)), 0.32337401293247753)]
    # innercircles = [Circle{2,T}(V((0.0473759, 0.0717498)), 0.3669856291902664), Circle{2,T}(V((1.20823, 0.146365)), 0.4936991855455834), Circle{2,T}(V((0.468667, 0.647647)), 0.14709330779994115), Circle{2,T}(V((-0.536947, 0.947319)), 0.40844964948634166), Circle{2,T}(V((0.829234, 1.11387)), 0.2712912144973267), Circle{2,T}(V((1.76985, 1.25328)), 0.4257224157797549), Circle{2,T}(V((0.237131, 1.01989)), 0.15490132984774477), Circle{2,T}(V((0.988679, 1.93488)), 0.33744120020697527), Circle{2,T}(V((0.294656, 1.566)), 0.23245752368812128), Circle{2,T}(V((-0.328417, 1.82669)), 0.25223173008733246)]
    # rect = Rectangle{2,T}(V((-0.382468, 0.172455)), V((1.37816, 1.93308)))
    
    allcircles = collect(Iterators.flatten(zip(outercircles, innercircles)))
    mincircdist = minimum_signed_edge_distance(outercircles)
    
    if PLOT
        p = plot(rect; aspectratio = :equal);
        for c in allcircles; plot!(p, c); end
        display(p)
    end

    # Signed distance function for many circles
    function dcircles(x::Vec{2,T}, cs::Vector{C}) where {T, C<:Circle{2}}
        d = T(dunion()) # initial value s.t. dunion(d, x) == x for all x
        @inbounds for i in eachindex(cs)
            d = dunion(d, dcircle(x, cs[i])) # union of all circle distances
        end
        return d
    end
    
    function dgrid(x::Vec{2,T}) where {T}
        douters, dinners = dcircles(x, outercircles), dcircles(x, innercircles)
        drect = drectangle0(x, rect)
        dext = ddiff(drect, douters)
        dtori = ddiff(douters, dinners)
        d = dintersect(drect, dunion(dext, dtori, dinners))
        return d
    end

    # Compute medial axis points
    h0 = mincircdist/2
    bbox = [xmin(rect) ymin(rect); xmax(rect) ymax(rect)]
    pfix = [V[corners(rect)...]; reduce(vcat, intersection_points(c,rect) for c in allcircles)]
    fd = dgrid

    # Subregion definitions. Order of iterator is important, as we want to project
    # outercircle points first, followed by inner circle points.
    # Also, zipping the circles together allows the comprehension to be well typed.
    fsubs = [x->dcircle(x,c) for c in allcircles]
    # fsubs = [x->dintersect(drectangle0(x,rect), dcircle(x,c)) for c in allcircles]
    
    # pmedial, _ = DistMesh.medial_axis_search(fd, h0, bbox)
    # fh = hgeom(fd, h0, bbox; alpha = alpha)
    
    # dmax = maximum(x -> abs(fd(x)), DistMesh.cartesian_grid_generator(bbox, h0/4))
    dmax = beta * btparams.R_mu
    # fh = x -> alpha + min(abs(fd(x))/dmax, one(eltype(x)))
    hallcircles = x -> min(abs(dcircles(x, outercircles)), abs(dcircles(x, innercircles)))/dmax
    fh = x -> alpha + min(hallcircles(x), huniform(x))

    return fd, fsubs, fh, h0, bbox, pfix, outercircles, innercircles, rect
end

using StatsPlots
plotargs = (size=(800,800), grid=false, legend=nothing, labels=nothing)
# plotly(;plotargs...) # html-based plotting in browser (nice, interactive, but slow)
# pyplot(;plotargs...) # interacting plotting gui (nice, but a bit slow)
gr(;plotargs...) # non-interactive plotting gui (fast, not interactive)

gamma, alpha, beta = 1.5, 0.5, 0.5
fd, fsubs, fh, h0, bbox, pfix, outercircles, innercircles, rect = runblocktorreygrid(
    Ncircles = 20,
    alpha = alpha/gamma,
    beta = beta,
    PLOT = true
);

p, t = kmg2d(fd, fsubs, fh, h0/gamma, bbox, 1, 0, pfix;
    QMIN = 0.3,
    MAXITERS = 1000,
    FIXPOINTSITERS = 250,
    FIXSUBSITERS = 200,
    VERBOSE = true,
    DETERMINISTIC = true,
    PLOT = false,
    PLOTLAST = true
);
savefig("tmp_grid.pdf")
gui()

# p0, t0 = kmg2d(fd, [], fh, h0/gamma, bbox, 1, 0, pfix;
#     QMIN = 0.5,
#     MAXITERS = 1000,
#     FIXPOINTSITERS = 250,
#     FIXSUBSITERS = 250,
#     VERBOSE = true,
#     DETERMINISTIC = true,
#     PLOT = false,
#     PLOTLAST = true
# );
# savefig("tmp_grid_initial.pdf")
# gui()

# p, t = kmg2d(fd, fsubs, fh, h0/gamma, bbox, 1, 0, pfix, p0;
#     QMIN = 0.3,
#     MAXITERS = 1000,
#     FIXPOINTSITERS = 50,
#     FIXSUBSITERS = 50,
#     DENSITYCTRLFREQ = 50,
#     VERBOSE = true,
#     DETERMINISTIC = true,
#     PLOT = false,
#     PLOTLAST = true
# );
# savefig("tmp_grid.pdf")
# gui()

nothing
