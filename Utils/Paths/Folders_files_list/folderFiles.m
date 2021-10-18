function fileNamesList=folderFiles(foldersList, nFolderDepth, filesFilter, flagGUI, browserTitle)
%% folderFiles
% Returns cell array of file names located under input folders.
%
%% Syntax:
%	fileNamesList=folderFiles;
%	fileNamesList=folderFiles(foldersList);
%	fileNamesList=folderFiles(foldersList, nFolderDepth);
%	fileNamesList=folderFiles(foldersList, nFolderDepth, filesFilter);
%	fileNamesList=folderFiles(foldersList, nFolderDepth, filesFilter, flagGUI);
%	fileNamesList=folderFiles(foldersList, nFolderDepth, filesFilter, flagGUI, browserTitle);
%
%% Description:
% This functions goal is to return a cell array of names of files located under user
%   defined folders. The input should be a cell array of parent directories. The  function
%   also supports input of a single directory name string. Absolute file path is used,
%	replacing the relative path.
%   The user can choose the files or directories including files using the OS explorer- by
%   enabling the 'flagGUI' input. I was somewhat un-pleased from the multiple similar
%   function were proposed and submitted to Matlab File exchange (no offence, with
%   greatest respect to the authors and their work):
%    -  http://www.mathworks.com/matlabcentral/fileexchange/index?term=tag%3A%22directories%22&sort=downloads_desc
%    -  http://www.mathworks.com/matlabcentral/fileexchange/index?term=tag%3A%22files%22&sort=downloads_desc
%    -  http://www.mathworks.com/matlabcentral/fileexchange/index?term=tag%3A%22dir%22&sort=downloads_desc
%   During my first programming course I was taught that code using recursion is a bad
%   code. Recursive code is hard to understand, develop and maintain. Changing folders or
%   Matlab path during run time is also a bad thing- it takes more time and can cause
%   unwanted effects in Matlab environment. Therefore I've written my implementation,
%   witch, I believe, has some advantages over methods proposed earlier:
%   - It is not recursive but iterative.
%   - It does not changes the current folder (not using cd command).
%   - It uses "ls" function, which is supposed to be faster then "dir" function.
%   - According to some measurements I've made it runs faster the the alternatives.
%   - I also  believe is is clearly written, so it should be easy to understand and maintain.
%   - It supports wildCards.
%
%% Input arguments (defaults exist):
%	foldersList-    a list of parent directories. A string with folder name, or a cell
%       array of multiple folder names.
%   nFolderDepth-   an integer specifying how deep should the function go over the sub
%       folders tree. This way "0" would return user inputs (echo), while "Inf", or
%       any value higher then number of folder tree levels would go over all sub folders.
%       All values inbetween , will return the approproaite list of files located in
%       user defined sub-folders hierarchy.
%   flagGUI-        When enabled allows the user to choose the foldersList folders
%       elements using the Explorer.
%   browserTitle-   a string presented in the folder browser menu.
%
%% Output arguments:
%	fileNamesList-    a cell array of found file names, with their absolute path.
%
%% Issues & Comments:
%
%% Example I:
%	fileNamesList=folderFiles(pwd, 0);
%   fprintf('File names (+path) in current directory\n');
%   fprintf('%s\n', fileNamesList{:});
%
%	fileNamesList=folderFiles(pwd, Inf);
%   fprintf('File names (+path) in all current directory sub-folders\n');
%   fprintf('%s\n', fileNamesList{:});
%
%% Example II:
%	wildCard='*.m';
%   fileNamesList=folderFiles(pwd, 0, wildCard);
%   fprintf('File names (+path) %s in current directory\n', wildCard);
%   fprintf('%s\n', fileNamesList{:});
%
%% See also:
%  - folderSubFolders
%  - dir
%  - ls
%
%% Revision history:
% First version: Nikolay S. 2013-03-10.
% Last update:   Nikolay S. 2013-03-11.
%
% *List of Changes:*
%

%% Default params values
if nargin < 4
    if nargin==0 % if no iputs supplied- force using explorer
        foldersList={};
    end
    if isempty(foldersList)
        flagGUI=true;
    else
        flagGUI=false; % by default, explorer will not be used
    end
end

if nargin < 2
    nFolderDepth=0;
end

if nargin < 3
    filesFilter={};
end

% when user did not specify flag value, but specified further parameters- enable browser
if isempty(foldersList) && isempty(flagGUI)
    % if user has explicitly set flagGUI=false,  it will remain false,
    % (good for preventing unwanted user promts)
    flagGUI=true;
end

if exist('browserTitle', 'var')~=1
    browserTitle='Select input files';
end

if exist('flagGUI', 'var')~=1 || isempty(flagGUI)
    flagGUI=false;
end

%% Program start
% get list of relevant sub-folders
foldersAgregator=folderSubFolders(foldersList, nFolderDepth, flagGUI, browserTitle, false);

fileNamesList={};
nFolders=length(foldersAgregator);

%% Find wildCards from the filesFilter
wildCard={};
if ischar(filesFilter)
    filesFilter={filesFilter};
end
if ~isempty(filesFilter)
    % move the wildCards from "filesFilter" to "wildCard"
    isWildCard=not( cellfun(@isempty, cellfun( @findstr, filesFilter,...
        repmat({'*'}, size(filesFilter)),  'UniformOutput', false )) );
    wildCard=filesFilter(isWildCard);
    filesFilter=filesFilter(~isWildCard);
end	% if ~isempty(filesFilter)

nWildCard=length(wildCard);
namesList={};
for iFolder=1:nFolders % list contents of all relevant subfolders files
    % For each relevant sub-folder- get the relevant files 
    currFolder=foldersAgregator{iFolder};
    if ~strcmpi(currFolder(end), filesep)
        % if there is no '/' at the end of folder name, add it for future use
        currFolder=strcat(currFolder, filesep);
    end
    
    %% Apply Wild Cards
    if nWildCard > 0 % if wildCards exist, run "ls" using them
        for iWildCard=1:nWildCard
            % this will get a list for each wildCard
            namesList=cat( 1, namesList,...
                cellstr(ls( strcat(currFolder, wildCard{iWildCard}) )) );
        end
        namesList=unique(namesList); % Disregard repeating files
    else
        namesList=cellstr( ls(foldersAgregator{iFolder}) );
    end % if nWildCard > 0
    namesList=strcat( currFolder, namesList );  % add path to have full folder names
    
    isFolderName=cellfun(@isdir, namesList);    % find directory names
    folderFiles=namesList(~isFolderName);       % ignore directory names, work on files
    
    if isempty(folderFiles) % empty folder
        continue;
    end % if ~isempty(folderFiles)
    
    if ~isempty(filesFilter) % Apply filesFilter, if exist
        for iFileFilter=1:numel(filesFilter)
            if not( strcmpi(filesFilter{iFileFilter}(1), '.') )
                % append filesFilter extensions with "." when needed
                filesFilter{iFileFilter}=strcat('.', filesFilter{iFileFilter});
            end
        end % for iFileFilter=1:numel(filesFilter)
        
        % Find files with relevant file extentions
        [~, ~, fileExt]=cellfun(@fileparts, folderFiles, 'UniformOutput', false);
        
        isLegalFileExt=false( size(folderFiles) ); % Allocate logical variable
        for iFile=1:numel(isLegalFileExt)
            isLegalFileExt(iFile)=any(strcmpi(fileExt{iFile}, filesFilter));
        end
        % ignore files with extentions different from those specified by filesFilter cell
        %   array list
        folderFiles=folderFiles(isLegalFileExt);
        
    end % if ~isempty(filesFilter)
    fileNamesList=cat(1, fileNamesList, folderFiles);
end	% for iFolder=1:nFolders % list contents of all relevant subfolders files