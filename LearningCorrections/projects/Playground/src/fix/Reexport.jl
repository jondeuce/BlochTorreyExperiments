module Reexport

"""
Verbatim copy of `@reexport` macro from `Reexport.jl` until v1.0 is tagged:

    https://github.com/simonster/Reexport.jl/pull/23
    https://github.com/simonster/Reexport.jl/blob/258a4088bb77ee4821bf2da1c73fd6e4897fd43c/src/Reexport.jl
"""
macro reexport(ex)
    isa(ex, Expr) && (ex.head == :module ||
                      ex.head == :using ||
                     (ex.head == :toplevel &&
                       all(e->isa(e, Expr) && e.head == :using, ex.args))) ||
        error("@reexport: syntax error")

    if ex.head == :module
        modules = Any[ex.args[2]]
        ex = Expr(:toplevel, ex, :(using .$(ex.args[2])))
    elseif ex.head == :using && all(e->isa(e, Symbol), ex.args)
        modules = Any[ex.args[end]]
    elseif ex.head == :using && ex.args[1].head == :(:)
        symbols = [e.args[end] for e in ex.args[1].args[2:end]]
        return esc(Expr(:toplevel, ex, :(eval(Expr(:export, $symbols...)))))
    else
        modules = Any[e.args[end] for e in ex.args]
    end

    esc(Expr(:toplevel, ex,
             [:(eval(Expr(:export, names($mod)...))) for mod in modules]...))
end

end # module Reexport
