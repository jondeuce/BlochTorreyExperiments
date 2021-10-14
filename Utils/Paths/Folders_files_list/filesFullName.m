function fullFileName=filesFullName(inFile, filesExtList, dlgTitle, isErrMode)
%% filesFullName
% The function attempts to find a full file name (path+file name+extention), filling the
%  missing gaps.
%
%% Syntax
%  fullFileName=filesFullName(inFile, filesExtList);
%  fullFileName=filesFullName(inFile);
%
%% Description
% The function uses Matlab build in commands "fileattrib" and "which" to get the file
%   details, one of which is the files full path we desire. Files of file-type (extention)
%   not suiting the user defined files extentions list (filesExtList) will be filtered
%   out.
%
%% Input arguments (defaults exist):
%  inFile- input file name. inFile must include file name. File path may be ommited, if
%     file is in Matlab path. File extention can be ommited for simplisuty or out of
%     laziness.
%  filesExtList- a list of extentions describing the file type we wish to detect. While
%     usually defaylu shold be empty- which wiill accpet all types of file, currently
%     default file types are Graphical or Videos. If all filesExtList elements are non
%     empty, only files with extention from the filesExtList list will be processed.All
%     the rest will get empty output.
%  dlgTitle- a string used in the files explorer menu.
%  isErrMode- a logical variable defining fnction behaviour in case of non existent file.
%     When enabled- an error messge will be issued (default behaviour). When disabled, an
%     empty name will be returned, without an error
%
%% Output arguments
%   fullFileName-  a full file name (path+file name+extention).
%
%% Issues & Comments
%
%% Example
% "fileattrib" command fails sometimes for an unknown reason, therefore slower "which"
%   command is used.
%
%% See also
% - folderSubFolders
%
%% Revision history
% First version: Nikolay S. 2012-05-01.
% Last update:   Nikolay S. 2013-04-21.
%
% *List of Changes:*
%
% 2013-04-21- filesExtList treatment changed. If all filesExtList are non empty, only
%   files with extention from the list will be processed. An empty output will be returned
%   otherwise.
% 2012-11-14- isErrMode: user can select how to react in case of missing file.
% 2012-07-31- dlgTitle: custom browser title added.
% 2012-07-19- Empty or missing input file name result in opening a browser
% 2012-05-21- Taken care of fileattrib error.
%
if nargin < 4
    isErrMode=true;
end
if nargin < 3
    dlgTitle='Select input file.';
end
if nargin < 2 || isempty(filesExtList)% if no filesExtList was provided try finding video or graphical files
    videoFormats= VideoReader.getFileFormats();
    videoExtList={videoFormats.Extension};    % video files extentions
    imageFormats=imformats;
    imageExtList=cat(2, imageFormats.ext);    % image files extentions
    filesExtListGUI=cat(2, videoExtList, imageExtList);
    filesExtList={};
else
    filesExtListGUI=filesExtList;
end

if nargin < 1
    inFile=[];
end

if isempty(inFile)
    filesFilter=sprintf('*.%s;', filesExtListGUI{:});
    
    [fileName, pathName, ~] = uigetfile(filesFilter, dlgTitle);
    if ischar(fileName) % single file was chosen
        fullFileName=strcat(pathName, fileName);
    else % cancel was pressed
        fullFileName=inFile;
    end
    return;
end % if isempty(inFile)

if ischar(filesExtList)
    filesExtList={filesExtList};
end

isAllFilesTypes=any( cellfun(@isempty, filesExtList) );
[~, ~, fileExt] = fileparts(inFile);
isEmptyFileExt=isempty(fileExt);
isFileExists=( exist(inFile, 'file')==2 );
if ~isFileExists || (isFileExists && isEmptyFileExt)
    % if no such file is found, or file found, but file extention was ommited by user
    if isErrMode
        % error if no file found for such an extention
        assert( isEmptyFileExt, 'No such file exists.' ); 
    end
    for iFileExt=1:length(filesExtList) 
        % if no file extention was mentioned, try finding one from supported video file extentions list
        candidateFile=strcat(inFile, '.', filesExtList{iFileExt});
        if exist(candidateFile, 'file')==2
            inFile=candidateFile;
            break;
        end
    end % for iFileExt=1:length(filesExtList)
    if exist(inFile, 'file')~=2
        if isErrMode
            % Issue an error if no file found for such an extention
            error('No such file exists.');
        else
            % If error mode is disabled, return empty spaces for non existent files
            fullFileName=[];
            return;
        end	% if isErrMode
    end	% if exist(inFile, 'file')~=2
end	% if exist(inFile, 'file')~=2

[~, ~, fileExt] = fileparts(inFile);
if ~any(strcmpi( fileExt(2:end), filesExtList )) && ~isAllFilesTypes
    % ignore files that do not match filesExtList
    fullFileName=[];
    return;
end
fullFileName=inFile;

[stats, currFileAttr]=fileattrib(fullFileName);
% sometimes fileattrib fails witout any explanation
% assert( stats && ~strcmpi(currFileAttr, 'Unknown error occurred.') );
if ( stats && ~strcmpi(currFileAttr, 'Unknown error occurred.') )
    fullFileName=currFileAttr.Name;
else
    % if file exists but fileattrib failed use which
    fullFileName=which(fullFileName);
end
