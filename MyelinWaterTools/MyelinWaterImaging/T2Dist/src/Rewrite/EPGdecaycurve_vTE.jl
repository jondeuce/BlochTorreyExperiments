"""
Computes the normalized echo decay curve for a MR spin echo sequence with the given parameters.

ETL: Echo train length (number of echos)
flip_angle: Angle of refocusing pulses (degrees)
TE: First Interecho time (seconds)
TE2: second Interecho time (seconds)
np: number of echoes at first echo time
T2: Transverse relaxation time (seconds)
T1: Longitudinal relaxation time (seconds)
refcon: Value of Refocusing Pulse Control Angle
"""
function EPGdecaycurve_vTE_work(T, ETL)
    M = zeros(Complex{T}, 3*ETL)
    decay_curve = zeros(T, ETL)
    return @ntuple(M, decay_curve)
end
EPGdecaycurve_vTE_work(ETL) = EPGdecaycurve_vTE_work(Float64, ETL)

EPGdecaycurve_vTE(ETL::Int, flip_angle::T, TE::T, T2::T, T1::T, refcon::T) where {T} =
    EPGdecaycurve_vTE!(EPGdecaycurve_vTE_work(T, ETL), ETL, flip_angle, TE, T2, T1, refcon)

function EPGdecaycurve_vTE!(work, ETL::Int, flip_angle::T, TE::T, T2::T, T1::T, refcon::T) where {T}
    # Unpack workspace
    @unpack M, decay_curve = work
    @assert ETL > 1 && length(M) == 3*ETL && length(decay_curve) == ETL
    M .= 0 # zero initial vector

    # Precompute compute element flip matrices and other
    E2, E1, E2_half = exp(-TE/T2), exp(-TE/T1), exp(-(TE/2)/T2)
    T1mat = element_flip_mat(flip_angle)
    T2mat = element_flip_mat(flip_angle * (refcon/180))
    T2mat_elements = (c_½α_sq = real(T2mat[1,1]), s_½α_sq = real(T2mat[2,1]), s_α_½ = -imag(T2mat[3,1]), s_α = -imag(T2mat[1,3]), c_α = real(T2mat[3,3]))

    # Initialize magnetization phase state vector (MPSV) and set all
    # magnetization in the F1 state.
    @inbounds M[1] = E2_half * sind(flip_angle/2)

    # Apply first refocusing pulse and get first echo amplitude
    _M = reinterpret(SVector{3,Complex{T}}, M) # View of M as vector of SVector's
    @inbounds _M[1] = T1mat * _M[1]
    @inbounds decay_curve[1] = E2_half * abs(M[2])

    # Apply relaxation matrix
    relaxmat_action!(M, ETL, E2, E1)

    # Perform flip-relax sequence ETL-1 times
    @timeit_debug TIMER "Flip-Relax Sequence" begin
        @inbounds for i = 2:ETL
            # Perform the flip
            # @timeit_debug TIMER "flipmat_action!" begin
            flipmat_action!(M, ETL, T2mat_elements)
            # end

            # Record the magnitude of the population of F1* as the echo amplitude
            # and allow for relaxation
            decay_curve[i] = E2_half * abs(M[2])

            # Allow time evolution of magnetization between pulses
            # @timeit_debug TIMER "relaxmat_action!" begin
            relaxmat_action!(M, ETL, E2, E1)
            # end
        end
    end

    return decay_curve
end

"""
Computes the action of the transition matrix that describes the effect of the refocusing pulse
on the magnetization phase state vector, as given by Hennig (1988), but corrected by Jones (1997).
"""
function flipmat_action_oddR!(M, num_states, T2mat_elements)
    @assert length(M) == 3*num_states

    # Optimized rotation matrix multiplication loop, specialized for the specific
    # form of `element_flip_mat`. The resulting generated code is ~10X faster, but
    # is exactly equivalent to the following simple loop:
    # 
    #   _M = reinterpret(SVector{3,Complex{T}}, M)
    #   @inbounds for i in 1:num_states
    #       _M[i] = T2mat * _M[i]
    #   end
    @unpack c_½α_sq, s_½α_sq, s_α_½, s_α, c_α = T2mat_elements
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

    return nothing
end

"""
Computes the action of the relaxation matrix that describes the time evolution of the
magnetization phase state vector after each refocusing pulse as described by Hennig (1988)
"""
function relaxmat_action_oddR!(M, num_states, E2, E1)
    @assert length(M) == 3*num_states

    # Optimized relaxation matrix loop
    @inbounds mprev = M[1]
    @inbounds M[1] = E2 * M[2] # F1* --> F1
    @inbounds for i in 3:3:3*num_states-3 # @simd made it slower!
        m0, m1, m2 = M[i], M[i+1], M[i+2]
        M[i-1] = E2 * m2 # F(n)* --> F(n-1)*
        M[i  ] = E1 * m0 # Z(n)  --> Z(n)
        M[i+1] = E2 * mprev # F(n)  --> F(n+1)
        mprev = m1
    end
    @inbounds M[end-1] = 0
    @inbounds M[end] *= E1

    return nothing
end
