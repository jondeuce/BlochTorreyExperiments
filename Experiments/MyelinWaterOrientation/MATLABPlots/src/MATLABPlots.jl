module MATLABPlots

using Reexport
using DistMesh
using MeshUtils
using JuAFEM
@reexport using MATLAB

export mxsimpplot

function mxsimpplot(p::AbstractMatrix, t::AbstractMatrix;
        newfigure = false,
        hold = false,
        xlim = nothing,
        ylim = nothing,
        axis = nothing,
        expr = Float64[],
        bcol = [0.8, 0.9, 1.0],
        icol = [0.0, 0.0, 0.0],
        nodes = 0.0,
        tris = 0.0,
        facecol = Float64[]
    )
    @assert size(p,2) == 2 && size(t,2) == 3

    newfigure && mxcall(:figure, 0)
    hold && mxcall(:hold, 0, "on")

    if !(isempty(p) || isempty(t))
        mxcall(:simpplot, 0,
            Matrix{Float64}(p), Matrix{Float64}(t),
            expr, bcol, icol, nodes, tris, facecol
        )
    end

    !(xlim == nothing) && mxcall(:xlim, 0, xlim)
    !(ylim == nothing) && mxcall(:ylim, 0, ylim)
    !(axis == nothing) && mxcall(:axis, 0, axis)

    return nothing
end

function mxsimpplot(
        p::AbstractVector{V},
        t::AbstractVector{NTuple{3,Int}};
        kwargs...
    ) where {V<:Vec{2}}
    mxsimpplot(DistMesh.to_mat(p), DistMesh.to_mat(t); kwargs...)
end

mxsimpplot(g::G; kwargs...) where {G <: Grid{2,3}} = mxsimpplot(nodematrix(g), cellmatrix(g); kwargs...)
mxsimpplot(gs::AbstractArray{G}; kwargs...) where {G <: Grid{2,3}} = mxsimpplot(nodecellmatrices(gs)...; kwargs...)

end # module MATLABPlots