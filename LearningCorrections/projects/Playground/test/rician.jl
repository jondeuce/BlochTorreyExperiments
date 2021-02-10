using Playground, Test

"Return wrapper around `f(xs...)` which takes log-arguments instead, i.e. `logf(logxs...) = f(map(logx -> exp.(logx), logxs)...)`"
with_log_args(f) = function with_log_args_inner(logxs...)
    xs = map(logx -> exp.(logx), logxs)
    f(xs...)
end

#### Test approximation values

for (ftrue, fapprox) in [
        (x -> lib._besselix0(x), lib._besselix0_cuda_unsafe),
        (x -> lib._besselix1(x), lib._besselix1_cuda_unsafe),
        (lib._laguerre½, lib._laguerre½_cuda_unsafe),
    ]
    @testset "$fapprox" begin
        for T in [Float32, Float64], logx in -3:3, sgn in [-1,1]
            x = sgn * T(exp(logx))
            @test isapprox(ftrue(x), fapprox(x); rtol = 1e-4, atol = 1e-6)
        end
    end
end

#### Test approximation gradients

for (ftrue, fapprox) in [
        (x -> lib._besselix0(x), lib._besselix0_cuda_unsafe),
        (x -> lib._besselix1(x), lib._besselix1_cuda_unsafe),
        # (lib._laguerre½, lib._laguerre½_cuda_unsafe),
    ]
    @testset "∇$fapprox" begin
        for T in [Float32, Float64], logx in -3:3, sgn in [-1,1]
            x = sgn * T(exp(logx))
            @test lib.gradcheck(
                with_log_args(fapprox), T(logx);
                extrapolate = false, backward = true, forward = true, verbose = true,
                rtol = 1e-3, atol = 1e-4,
            )
        end
    end
end

@testset "∇neglogL_rician_unsafe" begin
    logxs = -1:1
    for T in [Float32, Float64], logx in logxs, logν in logxs, logσ in logxs
        @test lib.gradcheck(
            with_log_args(lib.neglogL_rician_unsafe), T(logx), T(logν), T(logσ);
            extrapolate = false, backward = true, forward = true, verbose = true,
            rtol = 1e-3, atol = 1e-4,
        )
    end
end

#= _laguerre½
let
    vals = Dict{Int,BigFloat}(
        1       => big"1.44649134408317183335863878689608088636624386024796164711195852984091354675405604850815082040598215708783541465221",
        150     => big"13.8428182132579207630845698226782480139394357529419478259979332171281698876990601582933968786450293076448578693085",
        250     => big"17.8590913502406046821168814494005149903283070693484246034663777670678881083836244124944028011196109619555364318691",
        1000    => big"35.6914040595513768651551927899531513343638822880127244775680481255076084811577905488106037306888201625798303423912",
        1000000 => big"1128.37944919033960964972053998058678715837204482409899836336271985095168033327696330342629488845179765606231172765",
    )
    for (x,v) in sort(vals; by = first)
        println("\nx = $x:")
        for T in [Float64, Float32]
            display(abs(_laguerre½(T(-x)) - v) < 5 * eps(T(v)))
            # F = x == 150 ? 250 : x == 250 ? 10 : 2
            # display(abs(_laguerre½(T(-x)) - v))
            # display(eps(T(v)))
            # if x <= 150
            #     display(abs(_L½_bessel_kernel(T(x)) - v) < F * eps(T(v)))
            #     # display(abs(_L½_bessel_kernel(T(x)) - v))
            #     # display(eps(T(v)))
            # end
            # if x >= 150
            #     display(abs(_L½_series_kernel(T(x)) - v) < F * eps(T(v)))
            #     # display(abs(_L½_series_kernel(T(x)) - v))
            #     # display(eps(T(v)))
            # end
        end
    end
end;
=#

#= _logbesseli0
let
    vals = Dict{Int,BigFloat}(
        1       => big"0.235914358507178648689414846199951780553937315868442236933",
        75      => big"71.92399534542726979766263061737789216870198000778955948380",
        250     => big"246.3208320120570875328350064326308479165696536189728583253",
        1000    => big"995.6273088898694646714677644808475148830463306781655261607",
        1000000 => big"999992.1733063128132527062308001677706659748509246985475905",
    )
    for (x,v) in sort(vals; by = first)
        println("\nx = $x:")
        for T in [Float64, Float32]
            display(abs(_logbesseli0(T(x)) - v) < 5 * eps(T(v)))
            # display(abs(_logbesseli0(T(x)) - v))
            # display(eps(T(v)))
            # F = x == 1 ? 5 : 2
            # if x <= 75
            #     display(abs(_logI0_bessel_kernel(T(x)) - v) < F * eps(T(v)))
            #     # display(abs(_logI0_bessel_kernel(T(x)) - v))
            #     # display(eps(T(v)))
            # end
            # if x >= 75
            #     display(abs(_logI0_series_kernel(T(x)) - v) < F * eps(T(v)))
            #     # display(abs(_logI0_series_kernel(T(x)) - v))
            #     # display(eps(T(v)))
            # end
        end
    end
end;
=#

#= _logbesseli1
let
    vals = Dict{Int,BigFloat}(
        1       => big"-0.57064798749083128142317323666480514121275701003814227921",
        75      => big"71.91728368097706026635707716796022782798273695917775824800",
        250     => big"246.3188279973098207462626972558328961077394570824750455984",
        1000    => big"995.6268086396399849229481182362161219299624703016781038929",
        1000000 => big"999992.1733058128130027060016331886034188380540456910701362",
    )
    for (x,v) in sort(vals; by = first)
        println("\nx = $x:")
        for T in [Float64, Float32]
            display(abs(_logbesseli1(T(x)) - v) < 5 * eps(T(v)))
            # display(abs(_logbesseli1(T(x)) - v))
            # display(eps(T(v)))
            # F = x == 1 ? 5 : 2
            # if x <= 75
            #     display(abs(_logI1_bessel_kernel(T(x)) - v) < F * eps(T(v)))
            #     # display(abs(_logI1_bessel_kernel(T(x)) - v))
            #     # display(eps(T(v)))
            # end
            # if x >= 75
            #     display(abs(_logI1_series_kernel(T(x)) - v) < F * eps(T(v)))
            #     # display(abs(_logI1_series_kernel(T(x)) - v))
            #     # display(eps(T(v)))
            # end
        end
    end
end;
=#

#= _besseli1i0m1
let
    vals = Dict{Int,BigFloat}(
        1       => big"-0.55361003410346549295231820480735733022374685259961217713",
        50      => big"-0.01005103262150224740734407053531738330479593697870198617",
        75      => big"-0.00668919153535887090022356792413031251065008144225797039",
        250     => big"-0.00200200805042034549987616998460352026581291107756201664",
        1000    => big"-0.00050012512519571980108182565204402140552990313131753794",
        1000000 => big"-5.000001250001250001953129062510478547812614665076603e-7"
    )
    for (x,v) in sort(vals; by = first)
        for T in [Float64, Float32]
            println("\nx = $x ($T):")
            display(abs(_besseli1i0m1(T(x)) - v) < 25 * eps(T(v)))
            # display(abs(_besseli1i0m1(T(x)) - v))
            # display(eps(T(v)))
            # F = x == 50 ? 250 : x == 75 ? 10 : 2
            # if x <= 50
            #     display(abs(_I1I0m1_bessel_kernel(T(x)) - v) < F * eps(T(v)))
            #     # display(abs(_I1I0m1_bessel_kernel(T(x)) - v))
            #     # display(eps(T(v)))
            # end
            # if x >= 50
            #     display(abs(_I1I0m1_series_kernel(T(x)) - v) < F * eps(T(v)))
            #     # display(abs(_I1I0m1_series_kernel(T(x)) - v))
            #     # display(eps(T(v)))
            # end
        end
    end
end;
=#

#= _besseli2i0
let
    vals = Dict{Int,BigFloat}(
        1       => big"0.107220068206930985904636409614714660447493705199224354277",
        50      => big"0.960402041304860089896293762821412695332191837479148079447",
        75      => big"0.973511711774276236557339295144643475000284002171793545877",
        250     => big"0.992016016064403362763999009359876828162126503288620496133",
        1000    => big"0.998001000250250391439602163651304088042811059806262635075",
        1000000 => big"0.999998000001000000250000250000390625812502095709562522933"
    )
    for (x,v) in sort(vals; by = first)
        for T in [Float64, Float32]
            println("\nx = $x ($T):")
            display(abs(_besseli2i0(T(x)) - v) < 5 * eps(T(v)))
            # display(abs(_besseli2i0(T(x)) - v))
            # display(eps(T(v)))
            # F = x == 1 ? 5 : 2
            # if x <= 50
            #     display(abs(_I2I0_bessel_kernel(T(x)) - v) < F * eps(T(v)))
            #     # display(abs(_I2I0_bessel_kernel(T(x)) - v))
            #     # display(eps(T(v)))
            # end
            # if x >= 50
            #     display(abs(_I2I0_series_kernel(T(x)) - v) < F * eps(T(v)))
            #     # display(abs(_I2I0_series_kernel(T(x)) - v))
            #     # display(eps(T(v)))
            # end
        end
    end
end;
=#

#= (log)pdf
let
    σ = 0.23
    xs = range(0.0, 8.0; length = 500)
    for ν in [0.0, 0.5, 1.0, 2.0, 4.0]
        d = Rician(ν, σ)
        x = 8.0 #rand(Uniform(xs[1], xs[end]))
        @show log(pdf(d, x))
        @show logpdf(d, x)
        @assert log(pdf(d, x)) ≈ logpdf(d, x)
    end
end
=#

#= ∇logpdf
let
    ν, σ, x = 100*rand(), 100*rand(), 100*rand()
    d = Rician(ν, σ)
    ∇ = ∇logpdf(d, x)
    # δ = cbrt(eps())
    # ∇ν = (logpdf(Rician(ν + δ, σ), x) - logpdf(Rician(ν - δ, σ), x)) / 2δ
    # ∇σ = (logpdf(Rician(ν, σ + δ), x) - logpdf(Rician(ν, σ - δ), x)) / 2δ
    ∇ν = FiniteDifferences.central_fdm(3,1)(_ν -> logpdf(Rician(_ν, σ), x), ν)
    ∇σ = FiniteDifferences.central_fdm(3,1)(_σ -> logpdf(Rician(ν, _σ), x), σ)
    ∇δ = (∇ν = ∇ν, ∇σ = ∇σ)
    display(∇); display(values(∇δ))
    display(map((x,y) -> (x-y)/y, values(∇δ), values(∇)))
end;
=#

#= ∇²logpdf
let
    ν, σ, x = 100*rand(), 100*rand(), 100*rand()
    d = Rician(ν, σ)
    δ = eps(ν)^(1/4)
    ∇ν² = ∂²logpdf_∂ν²(d, x)
    # ∇νδ² = (logpdf(Rician(ν + δ, σ), x) - 2 * logpdf(Rician(ν, σ), x) + logpdf(Rician(ν - δ, σ), x)) / δ^2
    ∇νδ² = FiniteDifferences.central_fdm(5,2)(_ν -> logpdf(Rician(_ν, σ), x), ν)
    display(∇ν²)
    display(∇νδ²)
    display((∇ν² - ∇νδ²) / max(abs(∇ν²), abs(∇νδ²)))
end;
=#
