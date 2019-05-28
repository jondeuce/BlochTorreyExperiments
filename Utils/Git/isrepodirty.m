function [isdirty] = isrepodirty(reponame)
%ISREPODIRTY [isdirty] = isrepodirty(reponame)

reponame = finduniquefolder(reponame);
[~, status_string] = system(['git -C ', reponame, ' status --short']);
isdirty = ~isempty(status_string);

end

