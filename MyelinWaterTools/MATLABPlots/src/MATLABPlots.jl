module MATLABPlots

using GeometryUtils
using Reexport
@reexport using MATLAB
import Dates

export mxsimpplot, mxsimpgif

include("mxsimpplot.jl")

end # module MATLABPlots