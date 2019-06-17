using ApproxFun, LinearAlgebra, Expokit
using ApproxFun: Δ, ProductDomain

dx, dy = Interval(-1.0, 1.0), Interval(-1.0, 1.0)
d = ProductDomain(dx,dy)
f = Fun((x,y) -> exp(-10(x+0.3)^2-20(y-0.2)^2), d)
g = Fun((x,y) -> 0.0, ∂(d))
A = [Dirichlet(d); Laplacian(d) + 100I]    # Δ is an alias for Laplacian()
QR = qr(A)
@time u = QR \ [g; f]     # 4s for ~3k coefficients

d = ChebyshevInterval()^2
A = [Dirichlet(d); Laplacian(d)]
f = Fun((x,y)->real(exp(x+im*y)),∂(d))
QR = qr(A)
@time ApproxFun.resizedata!(QR,:,150)
@time u = \(QR, [f; 0.0]; tolerance = 1e-10)
