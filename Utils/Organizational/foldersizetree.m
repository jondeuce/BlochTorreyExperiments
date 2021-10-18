function [ total ] = foldersizetree( directory )
%FOLDERSIZETREE This just returns the sum of the directory you pass it and
% prints it's subfolders.
% 
% Ex:
% mydir(pwd)
% will return the folder size of the present working directory.

allpaths = genpath(directory);

names = strsplit(allpaths,';');
if isempty(names{end}); names = names(1:end-1); end

total=0;
if isempty(names)
    % nothing
else
    for i=1:length(names)
        disp(names{i})
        d=dir(names{i});
        bytes = sum([d.bytes]);
        disp(['Size: ' num2str(bytes)])
        total=total+bytes;
    end
end

disp(['The Total Sum is: ' num2str(total)])

%Thanks for the tip

end
