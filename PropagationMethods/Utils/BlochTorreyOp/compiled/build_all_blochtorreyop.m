% Build all compiled c/c++ files
try, build_BlochTorreyAction, catch e, warning(e.message), end
try, build_Laplacian, catch e, warning(e.message), end
try, build_SevenPointDifferenceMasked, catch e, warning(e.message), end
try, build_SevenPointStencil, catch e, warning(e.message), end
try, build_SevenPointStencilMasked, catch e, warning(e.message), end
try, build_SevenPointSumMasked, catch e, warning(e.message), end

% Run BlochTorreyOp tests
BlochTorreyOp.test;
