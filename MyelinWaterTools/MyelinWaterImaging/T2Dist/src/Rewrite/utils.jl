# Global timer object
const TIMER = TimerOutput()

tic() = time_ns()
toc(t) = (time_ns() - t)/1e9

function lsqnonneg!(work, C, d)
    NNLS.load!(work, C, d)
    NNLS.solve!(work)
    return work.x
end
lsqnonneg_work(C, d) = NNLS.NNLSWorkspace(C, d)
lsqnonneg(C, d) = lsqnonneg!(lsqnonneg_work(C, d), C, d)

function set_diag!(A::AbstractMatrix, val)
    @inbounds for i in 1:min(size(A)...)
        A[i,i] = val
    end
    A
end

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
    # Xs = X[1]:0.001:X[end]
    Ys = spl.(Xs)
    y, index = findmin(Ys)
    x = Xs[index]
    return @ntuple(x, y)
end

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
    Xs = 0.0:0.001:X[end]
    _, ind = findmin(abs.(spl.(Xs) .- value))
    return Xs[ind]
end