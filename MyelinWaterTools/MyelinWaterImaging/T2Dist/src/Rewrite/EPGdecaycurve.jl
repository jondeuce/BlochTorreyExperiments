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
    _M = reinterpret(SVector{3,ComplexF64}, M) # View of M as vector of SVector's

    # Initialize magnetization phase state vector (MPSV) and set all
    # magnetization in the F1 state.
    @inbounds M[1] = exp(-(TE/2)/T2) * sind(flip_angle/2)
    
    # Compute flip and relax matrix temporaries
    @unpack T1mat, T2mat = flip_matrices(flip_angle, ETL, refcon)
    
    # Apply first refocusing pulse and get first echo amplitude
    @inbounds _M[1] = T1mat * _M[1]
    decay_curve[1] = abs(M[2]) * exp(-(TE/2)/T2)
    
    # Apply relaxation matrix
    relaxmat_action!(work, ETL, TE, T2, T1)
    
    # Perform flip-relax sequence ETL-1 times
    for i = 2:ETL
        # Perform the flip
        @timeit TIMER "flipmat_action!" begin
            flipmat_action!(work, ETL, T2mat)
        end
        
        # Record the magnitude of the population of F1* as the echo amplitude
        # and allow for relaxation
        decay_curve[i] = abs(M[2]) * exp(-(TE/2)/T2)
        
        # Allow time evolution of magnetization between pulses
        @timeit TIMER "relaxmat_action!" begin
            relaxmat_action!(work, ETL, TE, T2, T1)
        end
    end

    return decay_curve
end

"""
Computes the action of the transition matrix that describes the effect of the refocusing pulse
on the magnetization phase state vector, as given by Hennig (1988), but corrected by Jones (1997).
"""
function flipmat_action!(work, num_states, T2mat)
    @unpack M = work
    @assert length(M) == 3*num_states

    _M = reinterpret(SVector{3,ComplexF64}, M)
    @inbounds @simd for i in 1:num_states
        _M[i] = T2mat * _M[i]
    end

    return M
end

function flip_matrices(alpha, num_pulses, refcon)
    # Create element flip matrices
    α = alpha
    ᾱ = α * (refcon/180)
    T1mat = @SMatrix[   cosd(α/2)^2     sind(α/2)^2  -im*sind(α);
                        sind(α/2)^2     cosd(α/2)^2   im*sind(α);
                    -im*sind(α)/2    im*sind(α)/2        cosd(α)]
    T2mat = @SMatrix[   cosd(ᾱ/2)^2     sind(ᾱ/2)^2  -im*sind(ᾱ);
                        sind(ᾱ/2)^2     cosd(ᾱ/2)^2   im*sind(ᾱ);
                    -im*sind(ᾱ)/2    im*sind(ᾱ)/2        cosd(ᾱ)]
    return @ntuple(T1mat, T2mat)
end

"""
Computes the action of the relaxation matrix that describes the time evolution of the
magnetization phase state vector after each refocusing pulse as described by Hennig (1988)
"""
function relaxmat_action!(work, num_states, te, t2, t1)
    @unpack M_tmp, M = work
    @assert length(M_tmp) == length(M) == 3*num_states
    
    E2, E1 = exp(-te/t2), exp(-te/t1)
    M_tmp[1] = E2 * M[2] # F1* --> F1
    @inbounds @simd for i in 3:3:3*num_states-3
        M_tmp[i-1] = E2 * M[i+2] # F(n)* --> F(n-1)*
        M_tmp[i  ] = E1 * M[i  ] # Z(n)  --> Z(n)
        M_tmp[i+1] = E2 * M[i-2] # F(n)  --> F(n+1)
    end
    M_tmp[end-1] = 0
    M_tmp[end] = E1 * M[end]
    M .= M_tmp

    return M
end
