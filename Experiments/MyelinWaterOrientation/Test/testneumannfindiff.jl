function lap!(
        l::AbstractArray{T,2},
        x::AbstractArray{T,2},
        hx = one(real(T)),
        hy = one(real(T)),
        m_int::AbstractArray{Bool,2} = falses(size(x)),
        m_ext::AbstractArray{Bool,2} = falses(size(x))
    ) where {T}

    @inline is_ext(i,j) = @inbounds m_ext[i,j]
    @inline is_int(i,j) = @inbounds m_int[i,j]
    @inline is_mye(i,j) = @inbounds !(m_int[i,j] || m_ext[i,j])

    Nx, Ny = size(x)
    @inbounds for j in 1:Ny
        for i in 1:Nx
            l[i,j] = if is_ext(i,j)
                dxB = (i == 1  || !is_ext(i-1,j  )) ? zero(T) : (x[i  ,j  ] - x[i-1,j  ])/hx
                dxF = (i == Nx || !is_ext(i+1,j  )) ? zero(T) : (x[i+1,j  ] - x[i  ,j  ])/hx
                dyB = (j == 1  || !is_ext(i  ,j-1)) ? zero(T) : (x[i  ,j  ] - x[i  ,j-1])/hy
                dyF = (j == Ny || !is_ext(i  ,j+1)) ? zero(T) : (x[i  ,j+1] - x[i  ,j  ])/hy
                (dxF - dxB)/hx + (dyF - dyB)/hy
            elseif is_int(i,j)
                dxB = (i == 1  || !is_int(i-1,j  )) ? zero(T) : (x[i  ,j  ] - x[i-1,j  ])/hx
                dxF = (i == Nx || !is_int(i+1,j  )) ? zero(T) : (x[i+1,j  ] - x[i  ,j  ])/hx
                dyB = (j == 1  || !is_int(i  ,j-1)) ? zero(T) : (x[i  ,j  ] - x[i  ,j-1])/hy
                dyF = (j == Ny || !is_int(i  ,j+1)) ? zero(T) : (x[i  ,j+1] - x[i  ,j  ])/hy
                (dxF - dxB)/hx + (dyF - dyB)/hy
            else # is_mye(i,j)
                dxB = (i == 1  || !is_mye(i-1,j  )) ? zero(T) : (x[i  ,j  ] - x[i-1,j  ])/hx
                dxF = (i == Nx || !is_mye(i+1,j  )) ? zero(T) : (x[i+1,j  ] - x[i  ,j  ])/hx
                dyB = (j == 1  || !is_mye(i  ,j-1)) ? zero(T) : (x[i  ,j  ] - x[i  ,j-1])/hy
                dyF = (j == Ny || !is_mye(i  ,j+1)) ? zero(T) : (x[i  ,j+1] - x[i  ,j  ])/hy
                (dxF - dxB)/hx + (dyF - dyB)/hy
            end
        end
    end

    return l
end

function lap(
        x::AbstractArray{T,2},
        hx = one(real(T)),
        hy = one(real(T)),
        m_int::AbstractArray{Bool,2} = falses(size(x)),
        m_ext::AbstractArray{Bool,2} = falses(size(x))
    ) where {T}
    return lap!(zeros(T,size(x)),x,hx,hy,m_int,m_ext)
end
