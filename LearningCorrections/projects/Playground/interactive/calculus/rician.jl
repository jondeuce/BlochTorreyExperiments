using TaylorSeries, SpecialFunctions, Printf

# I_0(x), I_1(x) expanded at x=0
I0_zero(x) = big"1" + x^2/big"4" + x^4/big"64" + x^6/big"2304" + x^8/big"147456" + x^10/big"14745600" + x^12/big"2123366400" + x^14/big"416179814400" + x^16/big"106542032486400" + x^18/big"34519618525593600"
I1_zero(x) = x/big"2" + x^3/big"16" + x^5/big"384" + x^7/big"18432" + x^9/big"1474560" + x^11/big"176947200" + x^13/big"29727129600" + x^15/big"6658877030400" + x^17/big"1917756584755200" + x^19/big"690392370511872000"
I2_zero(x) = x^2/big"8" + x^4/big"96" + x^6/big"3072" + x^8/big"184320" + x^10/big"17694720" + x^12/big"2477260800" + x^14/big"475634073600" + x^16/big"119859786547200" + x^18/big"38355131695104000"

# I_0(x), I_1(x) scaled by `√x e^-x` expanded at w=0 where w=1/x
I0_inf(w) = (big"1" + w * big"1"/big"8" + w^2 * big"9"/big"128" + w^3 * big"75"/big"1024" + w^4 * big"3675"/big"32768" + w^5 * big"59535"/big"262144" + w^6 * big"2401245"/big"4194304" + w^7 * big"57972915"/big"33554432" + w^8 * big"13043905875"/big"2147483648" + w^9 * big"418854310875"/big"17179869184" + w^10 * big"30241281245175"/big"274877906944" + w^11 * big"1212400457192925"/big"2199023255552" + w^12 * big"213786613951685775"/big"70368744177664") / sqrt(2 * big(π))
I1_inf(w) = (big"1" - w * big"3"/big"8" - w^2 * big"15"/big"128" - w^3 * big"105"/big"1024" - w^4 * big"4725"/big"32768" - w^5 * big"72765"/big"262144" - w^6 * big"2837835"/big"4194304" - w^7 * big"66891825"/big"33554432" - w^8 * big"14783093325"/big"2147483648" - w^9 * big"468131288625"/big"17179869184" - w^10 * big"33424574007825"/big"274877906944" - w^11 * big"1327867167401775"/big"2199023255552" - w^12 * big"232376754295310625"/big"70368744177664") / sqrt(2 * big(π))
I2_inf(w) = (big"1" - w * big"15"/big"8" + w^2 * big"105"/big"128" + w^3 * big"315"/big"1024" + w^4 * big"10395"/big"32768" + w^5 * big"135135"/big"262144" + w^6 * big"4729725"/big"4194304" + w^7 * big"103378275"/big"33554432" + w^8 * big"21606059475"/big"2147483648" + w^9 * big"655383804075"/big"17179869184" + w^10 * big"45221482481175"/big"274877906944" + w^11 * big"1747193641318125"/big"2199023255552" + w^12 * big"298770112665399375"/big"70368744177664") / sqrt(2 * big(π))

# r = I1_inf(w) / I0_inf(w)
# I1_inf(t)
# I1_zero(w)

let
    # w, = set_variables("w", order = 20)
    w = Taylor1([big"0", big"1"], 20)
    for (order, Iν) in [(0,I0_zero), (1,I1_zero), (2,I2_zero)]
        @info order
        f = Iν(w)
        show(f)
        println("")
        for o in order .+ (0:2:12)
            c = getcoeff(f, o) * big"3.75"^(o-order)
            display(c)
            # @printf("%.7e\n", Float32(c))
        end
    end
end

#= x = zero expansion
let
    # w, set_variables("w", order = 20)
    w = Taylor1([big"0", big"1"], 20)
    x0 = 0.1
    besseli(0.0,x0) - I0_zero(w)(x0)
    besseli(1.0,x0) - I1_zero(w)(x0)
    besseli(2.0,x0) - I2_zero(w)(x0)
end
=#

#= x = inf expansion
let
    # w, set_variables("w", order = 20)
    w = Taylor1([big"0", big"1"], 20)
    x0 = 100.0
    z0 = 1/x0
    exp(-x0)*sqrt(x0)*besseli(0.0,x0) - I0_inf(w)(z0)
    exp(-x0)*sqrt(x0)*besseli(1.0,x0) - I1_inf(w)(z0)
    exp(-x0)*sqrt(x0)*besseli(2.0,x0) - I2_inf(w)(z0)
end
=#
