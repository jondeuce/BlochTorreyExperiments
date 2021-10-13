function [ varargout ] = periodic_copy( src, dst, period, timeout, copynow )
%PERIODIC_COPY Periodically copy file/folder at location 'src' to location
%'dst' with period 'period' seconds. Folder is copied multiple times, into
%separate folders with names given by datestr(now,30)

prnt = @(c) fprintf('Copying file/folder... (copycount = %d)\n',c);
copycount = 0;

if copynow
    copyfile(src,[dst,'/',datestr(now,30)]);
    copycount = copycount + 1;
    prnt(copycount);
end

t0 = tic;
while toc(t0) < timeout
    pause(period);
    copyfile(src,[dst,'/',datestr(now,30)]);
    copycount = copycount + 1;
    prnt(copycount);
end

if nargout > 0
    varargout{1} = copycount;
end

end

