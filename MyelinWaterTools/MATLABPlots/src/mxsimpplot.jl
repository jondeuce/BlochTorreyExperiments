function mxsimpplot(p::AbstractMatrix, t::AbstractMatrix;
        newfigure = true,
        visible = false,
        hold = false,
        xlim = nothing,
        ylim = nothing,
        axis = nothing,
        caxis = nothing,
        expr = Float64[],
        bcol = [0.8, 0.9, 1.0],
        icol = [0.0, 0.0, 0.0],
        nodes = 0.0,
        tris = 0.0,
        facecol = Float64[]
    )
    @assert size(p,2) == 2 && size(t,2) == 3

    # Default figure is not visible, undocked, with full screen size
    newfigure && mxcall(:figure, 0,
        "visible", visible ? "on" : "off",
        "windowstyle", "normal",
        "units", "normalized",
        "outerposition", [0 0 1 1])
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
    !(caxis == nothing) && mxcall(:caxis, 0, caxis)

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