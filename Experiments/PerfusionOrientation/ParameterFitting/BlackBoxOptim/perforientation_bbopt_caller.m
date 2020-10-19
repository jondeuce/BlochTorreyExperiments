function [loss] = perforientation_bbopt_caller(x, statepath)

    if nargin < 2; statepath = pwd; end

    % Save a copy of this script in the directory of the caller
    % backupscript  = sprintf('%s__%s.m', datestr(now,30), mfilename);
    % currentscript = strcat(mfilename('fullpath'), '.m');
    % copyfile(currentscript, backupscript);

    t_Geom = tic;
    Geom = load(fullfile(statepath, 'Geom.mat'));
    Geom = Uncompress(Geom.Geom);
    display_toc_time(toc(t_Geom), 'load + uncompress geometry');

    t_ObjFunMaker = tic;
    ObjFunMaker = load(fullfile(statepath, 'ObjFunMaker.mat'));
    ObjFunMaker = ObjFunMaker.ObjFunMaker;
    display_toc_time(toc(t_ObjFunMaker), 'load objective function');

    t_loss = tic;
    loss = ObjFunMaker(x, Geom);
    display_toc_time(toc(t_loss), 'compute loss');

end
