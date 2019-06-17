function [hash] = getrepohash(reponame)
%ISREPODIRTY [hash] = getrepohash(reponame)

reponame = finduniquefolder(reponame);
[~, hash] = system(['git -C ', reponame, ' rev-parse HEAD']);

end

