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
    decay_curve = zeros(ETL)
    return @ntuple(M, decay_curve)
end

EPGdecaycurve(ETL, flip_angle, TE, T2, T1, refcon) =
    EPGdecaycurve!(EPGdecaycurve_work(ETL), ETL, flip_angle, TE, T2, T1, refcon)

function EPGdecaycurve!(work, ETL, flip_angle, TE, T2, T1, refcon)
    # Unpack workspace
    @unpack M, decay_curve = work
    @assert length(M) == 3*ETL
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
Element matrix for the effect of the refocusing pulse on the magnetization state vector
"""
element_flip_mat(α) =
    @SMatrix[   cosd(α/2)^2    sind(α/2)^2 -im*sind(α);
                sind(α/2)^2    cosd(α/2)^2  im*sind(α);
            -im*sind(α)/2   im*sind(α)/2       cosd(α)]

"""
Computes the action of the transition matrix that describes the effect of the refocusing pulse
on the magnetization phase state vector, as given by Hennig (1988), but corrected by Jones (1997).
"""
function flipmat_action!(work, num_states, T2mat)
    @unpack M = work
    @assert length(M) == 3*num_states

    # Optimized rotation matrix multiplication loop, specialized for the specific
    # form of `element_flip_mat`. The resulting generated code is ~10X faster, but
    # is exactly equivalent to the following simple loop:
    # 
    #   _M = reinterpret(SVector{3,ComplexF64}, M)
    #   @inbounds for i in 1:num_states
    #       _M[i] = T2mat * _M[i]
    #   end
    c_½α_sq, s_½α_sq =  real(T2mat[1,1]),  real(T2mat[2,1])
    s_α_½, s_α, c_α  = -imag(T2mat[3,1]), -imag(T2mat[1,3]), real(T2mat[3,3])
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

    return M
end

"""
Computes the action of the relaxation matrix that describes the time evolution of the
magnetization phase state vector after each refocusing pulse as described by Hennig (1988)
"""
function relaxmat_action!(work, num_states, E2, E1)
    @unpack M = work
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

    return M
end
