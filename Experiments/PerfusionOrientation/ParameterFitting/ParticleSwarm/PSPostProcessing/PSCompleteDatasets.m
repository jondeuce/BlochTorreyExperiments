%% Properties to copy exactly
CompleteResults = Results;

alpha_range = 2.5:5:87.5;
[alpha_range, dR2_Data, TE, VoxelSize, VoxelCenter, GridSize, BinCounts] = get_GRE_data(alpha_range);

Nalphas = numel(alpha_range);
Ngeoms = numel(CompleteResults.Geometries);

CompleteResults.xdata = alpha_range;
CompleteResults.dR2_Data = dR2_Data;
CompleteResults.alpha_range = alpha_range;

%% Properties to copy and the fill in missing
CompleteResults.TimePts = cell(Ngeoms, Nalphas);
CompleteResults.S_CA = cell(Ngeoms, Nalphas);
CompleteResults.S_noCA = cell(Ngeoms, Nalphas);

CompleteResults.dR2 = zeros(1,Nalphas);
CompleteResults.dR2_all = zeros(Ngeoms,Nalphas);

%% Copy completed results in
[~,DONE_alpha_indices,~] = intersect(alpha_range(:), Results.alpha_range(:));
TODO_alpha_indices = setdiff((1:Nalphas).', DONE_alpha_indices);

for jj = 1:numel(DONE_alpha_indices)
    idx = DONE_alpha_indices(jj);
    
    for ii = 1:Ngeoms
        CompleteResults.TimePts{ii,idx} = Results.TimePts(:,1,ii);
        CompleteResults.S_CA{ii,idx} = Results.S_CA(:,jj,ii);
        CompleteResults.S_noCA{ii,idx} = Results.S_noCA(:,jj,ii);
        
        CompleteResults.dR2_all(ii,idx) = Results.dR2_all(ii,jj);
    end
    
    CompleteResults.dR2(1,idx) = Results.dR2(1,jj);
end

%% Update input arguments
vec = @(x) x(:);
unitsum = @(x) x/sum(vec(x));

CompleteResults.args.xdata = alpha_range;
CompleteResults.args.dR2_Data = dR2_Data;
CompleteResults.args.Weights = unitsum(BinCounts);

CompleteResults.args.FigTypes = {'png', 'eps', 'pdf', 'fig'};
CompleteResults.args.DiaryFilename = '';

% check that the same normalized weights were used
assert( norm(vec(unitsum(CompleteResults.args.Weights(DONE_alpha_indices))) - vec(Results.args.Weights)) < 5*norm(eps(vec(Results.args.Weights))) );

%% Simulate remaining results

args = CompleteResults.args;
args.xdata = alpha_range(TODO_alpha_indices);
args.dR2_Data = dR2_Data(TODO_alpha_indices);
args.Weights = unitsum(BinCounts(TODO_alpha_indices));

[dR2, NewResults] = perforientation_fun(args);

%% Copy new results in
for jj = 1:numel(TODO_alpha_indices)
    idx = TODO_alpha_indices(jj);
    
    for ii = 1:Ngeoms
        CompleteResults.TimePts{ii,idx} = NewResults.TimePts(:,1,ii);
        CompleteResults.S_CA{ii,idx} = NewResults.S_CA(:,jj,ii);
        CompleteResults.S_noCA{ii,idx} = NewResults.S_noCA(:,jj,ii);
        
        CompleteResults.dR2_all(ii,idx) = NewResults.dR2_all(ii,jj);
    end
    
    CompleteResults.dR2(1,idx) = NewResults.dR2(1,jj);
end

%% Save CompleteResults
save([fname,'__Completed','.mat'], 'CompleteResults');
