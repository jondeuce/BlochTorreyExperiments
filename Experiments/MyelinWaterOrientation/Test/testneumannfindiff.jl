function lap!(
        l::AbstractArray{T,2},
        x::AbstractArray{T,2},
        m_ext::AbstractArray{Bool,2},
        m_int::AbstractArray{Bool,2},
        hx::T = one(T),
        hy::T = one(T)
    ) where {T}

    @inline is_ext(i,j) = m_ext[i,j]
    @inline is_int(i,j) = m_ext[i,j]
    @inline is_mye(i,j) = !(m_int[i,j] || m_ext[i,j])

    for j in 1:size(x,2)
        for i in 1:size(x,1)


        end
    end

end
