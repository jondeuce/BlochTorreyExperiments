####
#### Miscellaneous utils
####
ndigits(x) = ceil(Int, log10(x))
logrange(a::T, b::T, len::Int) where {T} = T(10) .^ range(log10(a), log10(b); length = len)
normcdf(x::T) where {T} = erfc(-x/sqrt(T(2)))/2
normccdf(x::T) where {T} = erfc(x/sqrt(T(2)))/2

function set_diag!(A::AbstractMatrix, val)
    @inbounds for i in 1:min(size(A)...)
        A[i,i] = val
    end
    return A
end

####
#### Timing utilities
####
const TIMER = TimerOutput() # Global timer object

tic() = time_ns()
toc(t) = (time_ns() - t)/1e9

function hour_min_sec(t)
    hour = floor(Int, t/3600)
    min = floor(Int, (t - 3600*hour)/60)
    sec = floor(Int, t - 3600*hour - 60*min)
    return @ntuple(hour, min, sec)
end

####
#### NNLS utilities
####
function lsqnonneg!(work, C, d)
    NNLS.load!(work, C, d)
    NNLS.solve!(work)
    return work.x
end
lsqnonneg_work(C, d) = NNLS.NNLSWorkspace(C, d)
lsqnonneg(C, d) = lsqnonneg!(lsqnonneg_work(C, d), C, d)

####
#### Spline minimization
####
function spline_opt(X::AbstractVector, Y::AbstractVector)
    @assert length(X) == length(Y) && length(X) > 1
    deg_spline = min(3, length(X)-1)
    spl = Spline1D(X, Y; k = deg_spline)

    function opt(a, b)
        res = Optim.optimize(x->spl(x), a, b, Optim.Brent())
        return Optim.minimizer(res), Optim.minimum(res)
    end

    knots = get_knots(spl)
    y, idx = findmin(spl.(knots))
    x = knots[idx]

    @inbounds for i = 1:length(knots)-1
        _x, _y = opt(knots[i], knots[i+1])
        (_y < y) && (x = _x; y = _y)
    end

    return @ntuple(x, y)
end

function spline_opt_brute(X::AbstractVector, Y::AbstractVector)
    @assert length(X) == length(Y) && length(X) > 1
    deg_spline = min(3, length(X)-1)
    spl = Spline1D(X, Y; k = deg_spline)
    Xs = range(X[1], X[end], length = 100_000)
    # Xs = X[1]:eltype(X)(0.001):X[end]
    Ys = spl.(Xs)
    y, index = findmin(Ys)
    x = Xs[index]
    return @ntuple(x, y)
end

####
#### Spline root-finding
####
function spline_root(X::AbstractVector, Y::AbstractVector, value = 0)
    @assert length(X) == length(Y) && length(X) > 1
    deg_spline = min(3, length(X)-1)
    spl = Spline1D(X, Y; k=deg_spline)
    return find_zero(x -> spl(x) - value, (X[1], X[end]), Bisection())
end

function spline_root_brute(X::AbstractVector, Y::AbstractVector, value = 0)
    @assert length(X) == length(Y) && length(X) > 1
    deg_spline = min(3, length(X)-1)
    spl = Spline1D(X, Y; k=deg_spline)
    Xs = range(X[1], X[end], length = 100_000)
    # Xs = Xs[1]:eltype(X)(0.001):X[end]
    _, ind = findmin(abs.(spl.(Xs) .- value))
    return Xs[ind]
end