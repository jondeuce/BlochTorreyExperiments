using MMDLearning, Test

for N in 1:4
    Ts = [A{T,N} for A in [Array, CUDA.CuArray] for T in [Float64, Float32]]
    for T1 in Ts, T2 in Ts
        x = rand_similar(T1, (1 for _ in 1:ndims(T1))...)
        y = rand_similar(T2, (1 for _ in 1:ndims(T2))...)
        z = arr_similar(x, y)
        @test typeof(z) == typeof(x)
    end
end
