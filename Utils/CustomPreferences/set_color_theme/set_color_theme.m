function set_color_theme(name, dark)
    
    if nargin < 2, dark = 1; end
    
    [base, flag] = get_base16(name);
    
    if flag, dark = 1; end
    
    set_color_theme_(base, dark);
    
end

function set_color_theme_(base, dark)
    
    if dark
        base.bg = base.x00;
        base.fg = base.x05;
        base.hl = base.x01;
        base.he = base.x02;
    else
        base.bg = base.x07;
        base.fg = base.x02;
        base.hl = base.x06;
        base.he = base.x05;
    end

    % Don't use system colors
    com.mathworks.services.Prefs.setBooleanPref('ColorsUseSystem', 0);
        
    
    % *********************************************************************
    % Desktop tool colors
    % *********************************************************************
    
    % Text
    com.mathworks.services.Prefs.setColorPref( ...
        'ColorsText', java.awt.Color.decode(base.fg));
    com.mathworks.services.ColorPrefs.notifyColorListeners('ColorsText');
    % Background
    com.mathworks.services.Prefs.setColorPref( ...
        'ColorsBackground', java.awt.Color.decode(base.bg));
    com.mathworks.services.ColorPrefs.notifyColorListeners('ColorsBackground');
    
    
    % *********************************************************************
    % MATLAB syntax hightlighting colors
    % *********************************************************************
    
    % Keywords
    com.mathworks.services.Prefs.setColorPref( ...
        'Colors_M_Keywords', java.awt.Color.decode(base.x0D));
    % Strings
    com.mathworks.services.Prefs.setColorPref( ...
        'Colors_M_Strings', java.awt.Color.decode(base.x0F));
    % System Commands
    com.mathworks.services.Prefs.setColorPref( ...
        'Colors_M_SystemCommands', java.awt.Color.decode(base.x0A));
    % Comments
    com.mathworks.services.Prefs.setColorPref( ...
        'Colors_M_Comments', java.awt.Color.decode(base.x03));
    % Unterminated Strings
    com.mathworks.services.Prefs.setColorPref( ...
        'Colors_M_UnterminatedStrings', java.awt.Color.decode(base.x09));
    % Syntax Errors
    com.mathworks.services.Prefs.setColorPref( ...
        'Colors_M_Errors', java.awt.Color.decode(base.x08));
    
    
    % *********************************************************************
    % MATLAB Command Window colors
    % *********************************************************************
    
    % Error Text
    com.mathworks.services.Prefs.setColorPref( ...
        'Color_CmdWinErrors', java.awt.Color.decode(base.x08));
    % Warning Text
    com.mathworks.services.Prefs.setColorPref( ...
        'Color_CmdWinWarnings', java.awt.Color.decode(base.x09));
    % Hyperlinks
    com.mathworks.services.Prefs.setColorPref( ...
        'Colors_HTML_HTMLLinks', java.awt.Color.decode(base.x0D));
    
    
    % *********************************************************************
    % Code analyzer colors
    % *********************************************************************
    
    % Warnings
    com.mathworks.services.Prefs.setColorPref( ...
        'Colors_M_Warnings', java.awt.Color.decode(base.x09));
    % Autofix highlight
    com.mathworks.services.Prefs.setColorPref( ...
        'ColorsMLintAutoFixBackground', java.awt.Color.decode(base.he));
    com.mathworks.services.ColorPrefs.notifyColorListeners( ...
        'ColorsMLintAutoFixBackground');
    
    
    % *********************************************************************
    % Variable and function colors
    % *********************************************************************
    
    % Automatically Highlight
    com.mathworks.services.Prefs.setColorPref( ...
        'Editor.VariableHighlighting.Color', java.awt.Color.decode(base.he));
    % Variables with shared scope
    com.mathworks.services.Prefs.setColorPref( ...
        'Editor.NonlocalVariableHighlighting.TextColor', ...
        java.awt.Color.decode(base.x0C));
    
    
    % *********************************************************************
    % Section display options
    % *********************************************************************
    
    % Highlight Sections
    com.mathworks.services.Prefs.setColorPref( ...
        'Editorhighlight-lines', java.awt.Color.decode(base.he));
    % Text limit line color
    com.mathworks.services.Prefs.setColorPref( ...
        'EditorRightTextLimitLineColor', java.awt.Color.decode(base.he));
    % Current line color
    com.mathworks.services.Prefs.setColorPref(...
        'Editorhighlight-caret-row-boolean-color', ...
        java.awt.Color.decode(base.hl));
    

    
    % Use autofix highlighting
    com.mathworks.services.Prefs.setBooleanPref(...
        'ColorsUseMLintAutoFixBackground', 1);    
    % Highlight current line
    com.mathworks.services.Prefs.setBooleanPref(...
        'Editorhighlight-caret-row-boolean', 1);
    
end
