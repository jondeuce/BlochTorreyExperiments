%% load in data
PSLoadResults;

%% Keep best NumToKeep iterations
NumToKeep = min(50, numel(Results));
for ii = vec(sorted_inds(NumToKeep+1:end)).'
    [pathstr,fname,ext] = fileparts(files(ii).name);
    fprintf('%s\n',fname)
    delete([fname,'.mat']);
    delete([fname,'.png']);
end

%% delete diary
diaryfile = dir('*.txt');
delete(diaryfile.name)
