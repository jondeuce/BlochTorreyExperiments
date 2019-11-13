tic() = time_ns()
toc(t) = (time_ns() - t)/1e9

function lsqnonneg!(work, C, d)
    NNLS.load!(work, C, d)
    NNLS.solve!(work)
    return work.x
end
lsqnonneg(C, d) = lsqnonneg!(NNLS.NNLSWorkspace(C, d), C, d)

function set_diag!(A::AbstractMatrix, mu)
    @inbounds for i in 1:min(size(A)...)
        A[i,i] = mu
    end
    A
end
