"""
Computes the normalized echo decay curve for a MR spin echo sequence with the given parameters.

ETL: Echo train length (number of echos)
flip_angle: Angle of refocusing pulses (degrees)
TE: Interecho time (seconds)
T2: Transverse relaxation time (seconds)
T1: Longitudinal relaxation time (seconds)
refcon: Value of Refocusing Pulse Control Angle
"""
function EPGdecaycurve_work(ETL)
    M = zeros(ComplexF64, 3*ETL)
    M_tmp = zeros(ComplexF64, 3*ETL)
    decay_curve = zeros(ETL)
    return @ntuple(M, M_tmp, decay_curve)
end

EPGdecaycurve(ETL, flip_angle, TE, T2, T1, refcon) =
    EPGdecaycurve!(EPGdecaycurve_work(ETL), ETL, flip_angle, TE, T2, T1, refcon)

function EPGdecaycurve!(work, ETL, flip_angle, TE, T2, T1, refcon)
    # Unpack workspace
    @unpack M, M_tmp, decay_curve = work
    @assert length(M_tmp) == length(M) == 3*ETL
    M .= 0 # Zero initial vector
    
    # Precompute compute element flip matrices and other
    E2, E1, E2_half = exp(-TE/T2), exp(-TE/T1), exp(-(TE/2)/T2)
    T1mat = element_flip_mat(flip_angle)
    T2mat = element_flip_mat(flip_angle * (refcon/180))
    
    # Initialize magnetization phase state vector (MPSV) and set all
    # magnetization in the F1 state.
    @inbounds M[1] = E2_half * sind(flip_angle/2)
    
    # Apply first refocusing pulse and get first echo amplitude
    _M = reinterpret(SVector{3,ComplexF64}, M) # View of M as vector of SVector's
    @inbounds _M[1] = T1mat * _M[1]
    @inbounds decay_curve[1] = abs(M[2]) * E2_half
    
    # Apply relaxation matrix
    relaxmat_action!(work, ETL, E2, E1)
    
    # Perform flip-relax sequence ETL-1 times
    @timeit_debug TIMER "Flip-Relax sequence" begin
        for i = 2:ETL
            # Perform the flip
            # @timeit_debug TIMER "flipmat_action!" begin
            flipmat_action!(work, ETL, T2mat)
            # end
            
            # Record the magnitude of the population of F1* as the echo amplitude
            # and allow for relaxation
            @inbounds decay_curve[i] = abs(M[2]) * E2_half
            
            # Allow time evolution of magnetization between pulses
            # @timeit_debug TIMER "relaxmat_action!" begin
            relaxmat_action!(work, ETL, E2, E1)
            # end
        end
    end

    return decay_curve
end

"""
Computes the action of the transition matrix that describes the effect of the refocusing pulse
on the magnetization phase state vector, as given by Hennig (1988), but corrected by Jones (1997).
"""
function flipmat_action!(work, num_states, T2mat, ::Val{VERSION} = Val(8)) where {VERSION}
    @unpack M, Mr, Mi = work
    @assert length(M) == 3*num_states

    _M = reinterpret(SVector{3,ComplexF64}, M)

    if VERSION == 1
        # Simple matrix-mul
        @inbounds for i in 1:num_states # removed @simd
            _M[i] = T2mat * _M[i]
        end
    elseif VERSION == 2
        # Separate real/imag matrix components
        c_½α_sq = real(T2mat[1,1])
        s_½α_sq = real(T2mat[2,1])
        s_α_½ = -imag(T2mat[3,1])
        s_α = -imag(T2mat[1,3])
        c_α = real(T2mat[3,3])
        @inbounds for i in 1:num_states # removed @simd
            m = _M[i]
            mx, my, mz = m[1], m[2], m[3]
            _mx = c_½α_sq * mx + s_½α_sq * my - im * (s_α * mz)
            _my = s_½α_sq * mx + c_½α_sq * my + im * (s_α * mz)
            _mz = im * (s_α_½ * (my - mx)) + c_α * mz
            _M[i] = SVector{3,ComplexF64}(_mx, _my, _mz)
        end
    elseif VERSION == 3
        # Compute real/imag components separately
        c_½α_sq = real(T2mat[1,1])
        s_½α_sq = real(T2mat[2,1])
        s_α_½ = -imag(T2mat[3,1])
        s_α = -imag(T2mat[1,3])
        c_α = real(T2mat[3,3])
        @inbounds for i in 1:num_states # removed @simd
            m = _M[i]
            mx_r, mx_i = real(m[1]), imag(m[1])
            my_r, my_i = real(m[2]), imag(m[2])
            mz_r, mz_i = real(m[3]), imag(m[3])
            s_α_mz_r = s_α * mz_r
            s_α_mz_i = s_α * mz_i
            _mx = Complex(c_½α_sq * mx_r + s_½α_sq * my_r + s_α_mz_i,
                          c_½α_sq * mx_i + s_½α_sq * my_i - s_α_mz_r)
            _my = Complex(s_½α_sq * mx_r + c_½α_sq * my_r - s_α_mz_i,
                          s_½α_sq * mx_i + c_½α_sq * my_i + s_α_mz_r)
            _mz = Complex(s_α_½ * (mx_i - my_i) + c_α * mz_r,
                          s_α_½ * (my_r - mx_r) + c_α * mz_i)
            _M[i] = SVector{3,ComplexF64}(_mx, _my, _mz)
        end
    elseif VERSION == 4
        # Same as V3 but with muladds
        c_½α_sq = real(T2mat[1,1])
        s_½α_sq = real(T2mat[2,1])
        s_α_½ = -imag(T2mat[3,1])
        s_α = -imag(T2mat[1,3])
        c_α = real(T2mat[3,3])
        @inbounds for i in 1:num_states # removed @simd
            m = _M[i]
            mx_r, mx_i = real(m[1]), imag(m[1])
            my_r, my_i = real(m[2]), imag(m[2])
            mz_r, mz_i = real(m[3]), imag(m[3])
            s_α_mz_r = s_α * mz_r; ms_α_mz_r = -s_α_mz_r
            s_α_mz_i = s_α * mz_i; ms_α_mz_i = -s_α_mz_i
            _mx = Complex(muladd(c_½α_sq, mx_r, muladd(s_½α_sq, my_r,  s_α_mz_i)),
                          muladd(c_½α_sq, mx_i, muladd(s_½α_sq, my_i, ms_α_mz_r)))
            _my = Complex(muladd(s_½α_sq, mx_r, muladd(c_½α_sq, my_r, ms_α_mz_i)),
                          muladd(s_½α_sq, mx_i, muladd(c_½α_sq, my_i,  s_α_mz_r)))
            _mz = Complex(muladd(s_α_½, mx_i - my_i, c_α * mz_r),
                          muladd(s_α_½, my_r - mx_r, c_α * mz_i))
            _M[i] = SVector{3,ComplexF64}(_mx, _my, _mz)
        end
    elseif VERSION == 5
        # Same as V2 but looping over originally complex vector
        c_½α_sq = real(T2mat[1,1])
        s_½α_sq = real(T2mat[2,1])
        s_α_½ = -imag(T2mat[3,1])
        s_α = -imag(T2mat[1,3])
        c_α = real(T2mat[3,3])
        @inbounds for i in 1:3:3*num_states # removed @simd
            mx, my, mz = M[i], M[i+1], M[i+2]
            M[i]   = c_½α_sq * mx + s_½α_sq * my - im * (s_α * mz)
            M[i+1] = s_½α_sq * mx + c_½α_sq * my + im * (s_α * mz)
            M[i+2] = im * (s_α_½ * (my - mx)) + c_α * mz
        end
    elseif VERSION == 6
        # Same as V3 but looping over originally complex vector
        c_½α_sq = real(T2mat[1,1])
        s_½α_sq = real(T2mat[2,1])
        s_α_½ = -imag(T2mat[3,1])
        s_α = -imag(T2mat[1,3])
        c_α = real(T2mat[3,3])
        @inbounds for i in 1:3:3*num_states # removed @simd
            mx, my, mz = M[i], M[i+1], M[i+2]
            mx_r, mx_i = real(mx), imag(mx)
            my_r, my_i = real(my), imag(my)
            mz_r, mz_i = real(mz), imag(mz)
            s_α_mz_r = s_α * mz_r
            s_α_mz_i = s_α * mz_i
            M[i]   = Complex(c_½α_sq * mx_r + s_½α_sq * my_r + s_α_mz_i,
                             c_½α_sq * mx_i + s_½α_sq * my_i - s_α_mz_r)
            M[i+1] = Complex(s_½α_sq * mx_r + c_½α_sq * my_r - s_α_mz_i,
                             s_½α_sq * mx_i + c_½α_sq * my_i + s_α_mz_r)
            M[i+2] = Complex(s_α_½ * (mx_i - my_i) + c_α * mz_r,
                             s_α_½ * (my_r - mx_r) + c_α * mz_i)
        end
    elseif VERSION == 7
        # Same as V4 but looping over originally complex vector
        c_½α_sq = real(T2mat[1,1])
        s_½α_sq = real(T2mat[2,1])
        s_α_½ = -imag(T2mat[3,1])
        s_α = -imag(T2mat[1,3])
        c_α = real(T2mat[3,3])
        @inbounds for i in 1:3:3*num_states # @simd made it slower!
            mx_r, mx_i = real(M[i  ]), imag(M[i  ])
            my_r, my_i = real(M[i+1]), imag(M[i+1])
            mz_r, mz_i = real(M[i+2]), imag(M[i+2])
            s_α_mz_r = s_α * mz_r
            s_α_mz_i = s_α * mz_i
            M[i]   = Complex(muladd(c_½α_sq, mx_r, muladd(s_½α_sq, my_r,  s_α_mz_i)),
                             muladd(c_½α_sq, mx_i, muladd(s_½α_sq, my_i, -s_α_mz_r)))
            M[i+1] = Complex(muladd(s_½α_sq, mx_r, muladd(c_½α_sq, my_r, -s_α_mz_i)),
                             muladd(s_½α_sq, mx_i, muladd(c_½α_sq, my_i,  s_α_mz_r)))
            M[i+2] = Complex(muladd(s_α_½, mx_i - my_i, c_α * mz_r),
                             muladd(s_α_½, my_r - mx_r, c_α * mz_i))
        end
    elseif VERSION == 8
        # Same as V7 but rearranged
        c_½α_sq = real(T2mat[1,1])
        s_½α_sq = real(T2mat[2,1])
        s_α_½ = -imag(T2mat[3,1])
        s_α = -imag(T2mat[1,3])
        c_α = real(T2mat[3,3])
        @inbounds for i in 1:3:3*num_states # @simd made it slower!
            mx_r, mx_i = real(M[i  ]), imag(M[i  ])
            my_r, my_i = real(M[i+1]), imag(M[i+1])
            mz_r, mz_i = real(M[i+2]), imag(M[i+2])
            s_α_mz_r = s_α * mz_r
            s_α_mz_i = s_α * mz_i
            M_r    = muladd(c_½α_sq, mx_r, muladd(s_½α_sq, my_r,  s_α_mz_i))
            M_i    = muladd(c_½α_sq, mx_i, muladd(s_½α_sq, my_i, -s_α_mz_r))
            M[i]   = Complex(M_r, M_i)
            M_r    = muladd(s_½α_sq, mx_r, muladd(c_½α_sq, my_r, -s_α_mz_i))
            M_i    = muladd(s_½α_sq, mx_i, muladd(c_½α_sq, my_i,  s_α_mz_r))
            M[i+1] = Complex(M_r, M_i)
            M_r    = muladd(s_α_½, mx_i - my_i, c_α * mz_r)
            M_i    = muladd(s_α_½, my_r - mx_r, c_α * mz_i)
            M[i+2] = Complex(M_r, M_i)
        end
    else # VERSION == 9
        # Same as V8 but real/imag M
        c_½α_sq = real(T2mat[1,1])
        s_½α_sq = real(T2mat[2,1])
        s_α_½ = -imag(T2mat[3,1])
        s_α = -imag(T2mat[1,3])
        c_α = real(T2mat[3,3])
        @inbounds for i in 1:3:3*num_states # @simd made it slower!
            mx_r = Mr[i]; my_r = Mr[i+1]; mz_r = Mr[i+2];
            mx_i = Mi[i]; my_i = Mi[i+1]; mz_i = Mi[i+2];
            s_α_mz_r = s_α * mz_r
            s_α_mz_i = s_α * mz_i
            Mr[i]   = muladd(c_½α_sq, mx_r, muladd(s_½α_sq, my_r,  s_α_mz_i))
            Mr[i+1] = muladd(s_½α_sq, mx_r, muladd(c_½α_sq, my_r, -s_α_mz_i))
            Mr[i+2] = muladd(s_α_½, mx_i - my_i, c_α * mz_r)
            Mi[i]   = muladd(c_½α_sq, mx_i, muladd(s_½α_sq, my_i, -s_α_mz_r))
            Mi[i+1] = muladd(s_½α_sq, mx_i, muladd(c_½α_sq, my_i,  s_α_mz_r))
            Mi[i+2] = muladd(s_α_½, my_r - mx_r, c_α * mz_i)
        end
    end

    return M
end

element_flip_mat(α) =
    @SMatrix[   cosd(α/2)^2    sind(α/2)^2 -im*sind(α);
                sind(α/2)^2    cosd(α/2)^2  im*sind(α);
            -im*sind(α)/2   im*sind(α)/2       cosd(α)]

"""
Computes the action of the relaxation matrix that describes the time evolution of the
magnetization phase state vector after each refocusing pulse as described by Hennig (1988)
"""
function relaxmat_action!(work, num_states, E2, E1, ::Val{VERSION} = Val(2)) where {VERSION}
    @unpack M_tmp, M = work
    @assert length(M_tmp) == length(M) == 3*num_states
    
    if VERSION == 1
        @inbounds M_tmp[1] = E2 * M[2] # F1* --> F1
        @inbounds @simd for i in 3:3:3*num_states-3
            M_tmp[i-1] = E2 * M[i+2] # F(n)* --> F(n-1)*
            M_tmp[i  ] = E1 * M[i  ] # Z(n)  --> Z(n)
            M_tmp[i+1] = E2 * M[i-2] # F(n)  --> F(n+1)
        end
        @inbounds M_tmp[end-1] = 0
        @inbounds M_tmp[end] = E1 * M[end]
        @inbounds M .= M_tmp
    elseif VERSION == 2
        @inbounds mprev = M[1]
        @inbounds M[1] = E2 * M[2] # F1* --> F1
        @inbounds for i in 2:3:3*num_states-4 # @simd made it slower!
            m1, m2, m3 = M[i+1], M[i+2], M[i+3]
            M[i] = E2 * m3 # F(n)* --> F(n-1)*
            M[i+1] = E1 * m1 # Z(n)  --> Z(n)
            M[i+2] = E2 * mprev # F(n)  --> F(n+1)
            mprev = m2
        end
        @inbounds M[end-1] = 0
        @inbounds M[end] = E1 * M[end]
    end

    return M
end
