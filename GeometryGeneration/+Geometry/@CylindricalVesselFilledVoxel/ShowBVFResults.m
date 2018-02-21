function ShowBVFResults( G )
%BVFRESULTS Display resulting BVF fractions compared to target BVF fracs.

relerr = @(x,y) (x-y)./max(abs(x),abs(y));

fprintf( '\n' );
fprintf( 'Result Minor BVF %%:  %11.8f %%\n', 100 * G.iBVF );
fprintf( 'Target Minor BVF %%:  %11.8f %%\n', 100 * G.Targets.iBVF );
fprintf(  'Minor Relative Err:  %11.8f   \n', relerr(G.iBVF,G.Targets.iBVF) );
fprintf( '\n' );

fprintf( 'Result Major BVF %%:  %11.8f %%\n', 100 * G.aBVF );
fprintf( 'Target Major BVF %%:  %11.8f %%\n', 100 * G.Targets.aBVF );
fprintf(  'Major Relative Err:  %11.8f   \n', relerr(G.aBVF,G.Targets.aBVF) );
fprintf( '\n' );

fprintf( 'Result Total BVF %%:  %11.8f %%\n', 100 * G.BVF);
fprintf( 'Target Total BVF %%:  %11.8f %%\n', 100 * G.Targets.BVF );
fprintf(  'Total Relative Err:  %11.8f   \n', relerr(G.BVF,G.Targets.BVF) );
fprintf( '\n' );

end

