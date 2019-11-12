tic() = time_ns()
toc(t) = (time_ns() - t)/1e9

function lsqnonneg!(work, C, d)
    load!(work, C, d)
    solve!(work)
    return work.x
end
lsqnonneg(C, d) = lsqnonneg!(NNLSWorkspace(C, d), C, d)
