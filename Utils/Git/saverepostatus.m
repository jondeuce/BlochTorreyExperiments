function [status_string] = saverepostatus(reponame)
%ISREPODIRTY [hash] = saverepostatus(reponame)

fullreponame = finduniquefolder(reponame);
fid = fopen([datestr(now,30), '__', reponame, '_repo_status.txt'], 'w');

if isrepodirty(reponame)
    warning('Repository %s is dirty!', reponame);
    fprintf(fid, '*****************************\n');
    fprintf(fid, '**** REPOSITORY IS DIRTY ****\n');
    fprintf(fid, '*****************************\n\n');
end

% Save local repo path
fprintf(fid, 'Local repository path:\n%s\n\n', fullreponame);

% Save current commit hash
fprintf(fid, 'Current commit hash:\n%s\n', getrepohash(fullreponame));

% Save current status
fprintf(fid, 'Current status:\n%s\n', getrepostatus(fullreponame));

fclose(fid);

end