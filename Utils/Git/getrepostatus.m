function [status_string] = getrepostatus(reponame)
%ISREPODIRTY [status_string] = getrepostatus(reponame)

reponame = finduniquefolder(reponame);
[~, status_string] = system(['git -C ', reponame, ' status --porcelain --verbose --untracked-files=all']);

end
