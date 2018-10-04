
expmv(t, A, b; kwargs...) = (f = similar(b); return expmv!(f, t, A, b; kwargs...))

function expmv!(f, t, A, b; kwargs...)
    f, _b, Ab = copyto!(f, b), copy(b), similar(b)
    _expmv!(f, t, A, _b, Ab; kwargs...) # `f`, `_b`, and `Ab` all modified in place
    return f
end

#NOTE: Assumes the following about the three input vectors, all three of which
# will be modified in place:
#   b is the initial vector (will be overwritten)
#   f is initialized such that all f[i] == b[i] (will be overwritten with result)
#   Ab is a buffer (will be overwritten)
function _expmv!(f, t, A, b, Ab = similar(b);
                 M = [],
                 prec = "double",
                 shift = true,
                 full_term = false,
                 prnt = false,
                 norm = LinearAlgebra.norm,
                 opnorm = LinearAlgebra.opnorm)
               # bal = false,

    #EXPMV   Matrix exponential times vector or matrix.
    #   [F,S,M,MV,MVD] = EXPMV(t,A,B,[],PREC) computes EXPM(t*A)*B without
    #   explicitly forming EXPM(t*A). PREC is the required accuracy, 'double',
    #   'single' or 'half', and defaults to CLASS(A).
    #
    #   A total of MV products with A or A^* are used, of which MVD are
    #   for norm estimation.
    #
    #   The full syntax is
    #
    #     [f,s,m,mv,mvd,unA] = expmv(t,A,b,M,prec,shift,bal,full_term,prnt).
    #
    #   unA = 1 if the alpha_p were used instead of norm(A).
    #
    #   If repeated invocation of EXPMV is required for several values of t
    #   or B, it is recommended to provide M as an external parameter as
    #   M = SELECT_TAYLOR_DEGREE(A,m_max,p_max,prec,shift,bal,true).
    #   This also allows choosing different m_max and p_max.

    #   Reference: A. H. Al-Mohy and N. J. Higham, Computing the action of
    #   the matrix exponential, with an application to exponential
    #   integrators. MIMS EPrint 2010.30, The University of Manchester, 2010.

    #   Awad H. Al-Mohy and Nicholas J. Higham, October 26, 2010.

    # if bal
    #     [D,B] = balance(A)
    #     if norm(B,1) < norm(A,1)
    #         A = B
    #         b = D\b
    #     else
    #         bal = false
    #     end
    # end

    n = size(A,1)
    T = promote_type(eltype(f), eltype(A), eltype(b))

    if shift
        mu = tr(A)/n
        A = A - mu*I
    end

    if isempty(M)
        tt = one(T)
        (M,mvd,alpha,unA) = select_taylor_degree(t*A, b; opnorm = opnorm)
        mv = mvd
    else
        tt = t
        mv = 0
        mvd = 0
    end

    tol = if prec == "double"
        T(2)^(-53)
    elseif prec == "single"
        T(2)^(-24)
    elseif prec == "half"
        T(2)^(-10)
    end

    s = 1

    if t == 0
        m = 0
    else
        (m_max,p) = size(M)
        U = diagm(0 => 1:m_max)
        C = (ceil.(abs(tt)*M))' * U
        zero_els = (LinearIndices(C))[findall(x->x==0, C)] #TODO (?)
        for el in zero_els
            C[el] = Inf
        end
        if p > 1
            cost, m = findmin(minimum(C, dims=1)[:]) # cost is the overall cost.
        else
            cost, m = findmin(C)  # when C is one column. Happens if p_max = 2.
        end
        if cost == Inf
            cost = 0
        end
        s = max(cost/m,1)
    end

    eta = one(T)
    shift && (eta = exp(t*mu/s))

    c1 = c2 = eltype(b)(Inf)
    for i = 1:s
        !full_term && (c1 = norm(b,Inf)) # only need to update if !full_term
        for k = 1:m
            prnt && println("$i/$s, $k/$m")
            # Next two lines replace `b = (t/(s*k))*(A*b)`
            mul!(Ab, A, b)
            mv = mv + 1
            b .= (t/(s*k)) .* Ab
            f .= f .+ b
            if !full_term
                c2 = norm(b,Inf) # only need to update if !full_term
                (c1 + c2 <= tol*norm(f,Inf)) && break
                c1 = c2
            end
        end
        shift && (f .= eta .* f)
        copyto!(b, f)
    end

    #if bal
    #    f = D*f
    #end

    #return (f,s,m,mv,mvd,unA)
    return f
end
