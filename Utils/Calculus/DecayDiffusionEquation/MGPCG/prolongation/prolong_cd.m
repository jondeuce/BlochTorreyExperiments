function [ y ] = prolong_cd( x2, x )
%PROLONG_CD [ y ] = prolong_cd( x2, x )

currentpath = cd;
cd(fileparts(mfilename('fullpath')));
build_prolong;
cd(currentpath);

[ y ] = prolong_cd( x2, x );

end

