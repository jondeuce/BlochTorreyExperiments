function subFoldersList=folderSubFolders(foldersList, nFolderDepth, flagGUI, browserTitle, isSortABC)
%% folderSubFolders
% Returns cell array of folder names located under input list (cell array) of folders.
%  
%% Syntax:
%     subFoldersList=folderSubFolders;
%     subFoldersList=folderSubFolders(foldersList);
%     subFoldersList=folderSubFolders(foldersList, nFolderDepth);
%     subFoldersList=folderSubFolders(foldersList, nFolderDepth, flagGUI);
%     subFoldersList=folderSubFolders(foldersList, nFolderDepth, flagGUI, browserTitle);
%     subFoldersList=folderSubFolders(foldersList, nFolderDepth, flagGUI, browserTitle, isSortABC);
%  
%% Description:
% This functions goal is to return a list (cell array) of names of sub-folders located 
%   under user defined folders list. The input should be a a cell array of parent
%   directories names. The function also supports input of a single directory name string.
%   Alternatively, the user can choose the folders using the OS explorer- by enabling the
%   'flagGUI' input.
%   Function properties:
%       - It is not recursive but iterative.
%       - It does not change the current folder (not using "cd" command).
%       - It uses "ls" function, which is supposed to be faster than "dir" function.
%       - According to some measurements I've made it runs faster than the alternatives. 
%       - I also believe it is clearly written and documented, so it should be easy to
%           understand and maintain.
%  
%% Input arguments (defaults exist):
%	foldersList-    a path to the parent directory. A string with a single folder name, or
%       a cell array of folder names.
%   nFolderDepth-   an integer specifying how deep should the function go over the sub
%       folders tree. This way "0" would return user inputs (echo), while "Inf", or
%       any value higher then number of folder tree levels would go over all existing sub
%       folders.
%   flagGUI-        When enabled allows the user to choose the folders using the folder
%                   browser (explorer in case of Windows OS).
%   browserTitle-   a string presented in the folder browser menu.
%   isSortABC-     	a logical flag allowing sorting the folders alphabetically when 
%                   enabled. When disabled, folders will be ordered by their "folder
%                   floor".
%  
%% Output arguments:
%     subFoldersList-    a cell array of found file names, with absolute path.
%
%% Issues & Comments:
%   None.
%
%% Example:
%	foldersList=folderSubFolders(pwd, 1);
%   fprintf('Folder names (+path) under current directory\n');
%   fprintf('%s\n', foldersList{:});
%
%	foldersList=folderSubFolders(pwd, Inf);
%   fprintf('All sub folders ( full path) in current directory.\n');
%   fprintf('%s\n', foldersList{:});
%
%
%% See also:
%   - filesFullName
%   - levelFolderName
%   - ls
%   - dir 
%
%% Revision history:
% First version: Nikolay S. 2013-03-10.
% Last update:   Nikolay S. 2013-04-10.
%
% *List of Changes:*
% 2013-04-10- folderFullPath is used instead of the same code inside this function

%% Default params values
if nargin < 4
    isSortABC=[];
end

if nargin < 3
    if nargin==0 % if no iputs supplied- force using explorer
        foldersList={};
    end
    if isempty(foldersList)
        flagGUI=true;
    else
        flagGUI=false; % by default, explorer will not be used
    end
end

if nargin < 2 % if nFolderDepth is not specified, search one folder level
    nFolderDepth=1;
end

% when user did nor specify flag value, but specified further parameters- enable browser
if isempty(foldersList) && isempty(flagGUI)
    % if user has explicitly set flagGUI=false,  it will remain false,
    % (good for preventing unwanted user promts)
    flagGUI=true;
end

if exist('browserTitle', 'var')~=1
    browserTitle='Select input files';
end

if exist('isSortABC', 'var')~=1 || isempty(isSortABC)
    isSortABC=false;
end

if exist('flagGUI', 'var')~=1 || isempty(flagGUI)
    flagGUI=false;
end

if (flagGUI)
    %% Select the directories/files using the Explorer
    if nargin < 2
        nFolderDepthUserChoise = questdlg('Please choose folders depth:', browserTitle,...
            '0-Current folder files', 'Custom value', 'Inf-All sub-folders files',...
            'Inf-All sub-folders files');
        switch(nFolderDepthUserChoise)
            case('0-Current folder files')
                nFolderDepth=0;
            case('Custom value')
                inputdlgPrompt='Enter folders depth value';
                inputdlgTitle='Custom folders depth';
                inputStr=inputdlg( inputdlgPrompt, inputdlgTitle, 1, {'1'} );
                nFolderDepth=round( str2double(inputStr) );
            otherwise
                nFolderDepth=Inf;
        end     % switch(nFolderDepthUserChoise)
    end     % if nargin < 2
    
    prevChosenFolder=pwd;
    anotherDir='More';
    while ~strcmpi(anotherDir, 'Finish')
        explStartDir = uigetdir(prevChosenFolder, browserTitle);
        
        if ~isequal(explStartDir, 0) % If cancel was not pressed
            % store last opened directory, to start with it on next Explorer use.
            prevChosenFolder=explStartDir; 
            foldersList=cat( 1, foldersList, {explStartDir} );
        end
               
        anotherDir = questdlg({'Need to choose another folder?',...
            'Press ''More'', to choose another folder.',...
            'Press ''Finish'' to finish choosing folders.'},...
            'Input folders selection',...
            'More', 'Finish', 'Finish');
    end     % while ~strcmpi(anotherDir,'Finish')
end     % if (flagGUI)

if ischar(foldersList) % convert string to cell aray
    foldersList={foldersList};
end

%% prepare foldersList elements folders for further processing
% get foldersList elements full folders path instead of relative/partial path if one was
% used
% foldersList=cellfun(@folderFullPath, foldersList, 'UniformOutput', false);
foldersList=cellfun(@GetFullPath, foldersList, 'UniformOutput', false);

%% Search for sub-folders
subFoldersList=foldersList;
prevFoldersAgregator=foldersList;
infdepthwarning_id = 'MATLAB:warn_truncate_for_loop_index';
warning('off',infdepthwarning_id); % ignore: "FOR loop index is too large. Truncating to 9223372036854775807."
for iFolderLevel=0:(nFolderDepth-1)  % Scan folders nFolderDepth 
    nCurrFolders=length(prevFoldersAgregator); % number of sub-folders in current level
    currFoldersAgregator={};
    for iSubFolder=1:nCurrFolders % go through current sub folders
        % find their subfolders
        currFolders=getFolderSubFolders( prevFoldersAgregator{iSubFolder} ); 
        if ~isempty(currFolders)
            if ~isequal( size(subFoldersList, 1), numel(subFoldersList) )
                subFoldersList=reshape(subFoldersList, [], 1);
            end
            if ~isequal( size(currFolders, 1), numel(currFolders) )
                currFolders=reshape(currFolders, [], 1);
            end

            subFoldersList=cat( 1, subFoldersList, currFolders );
            currFoldersAgregator=cat( 1, currFoldersAgregator, currFolders );
        end     % if ~isempty(currFolders)
    end     % for iFolder=1:nCurrFolders
    
    if isempty(currFoldersAgregator) % isequal(prevFoldersAgregator, currFoldersAgregator)
        break;
    end
    
    % update list of relevant folders for current iFolderLevel
    prevFoldersAgregator=currFoldersAgregator; 
end     % for iFolderLevel=0:(nFolderDepth-1)  % Scan folders nFolderDepth 
warning('on',infdepthwarning_id); %set warning back on

if isSortABC
   subFoldersList=sort( subFoldersList ); 
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                       Servise sub function                               %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function folderSubFolders=getFolderSubFolders( currFolder )
%% Find current folder sub folders
namesList=cellstr( ls(currFolder) );
folderSubFolders=strcat( currFolder, filesep, namesList );

% Remove non-folder elements- this can be ommited, but it will increase run time...
isFolderName=cellfun(@isdir, folderSubFolders);
folderSubFolders=folderSubFolders(isFolderName);

% Get each folder names elements: path, name extention.
[~, folderName, folderExt]=cellfun(@fileparts, folderSubFolders, 'UniformOutput', false);

% Folder must have a non empty name, and an empty extention
isFolder=not(cellfun( @isempty, folderName)) & cellfun( @isempty, folderExt);
% Remove names that are not legal folder names
folderSubFolders=folderSubFolders(isFolder);