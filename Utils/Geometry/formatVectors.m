function [ InDimsInv, OutDimsInv, PermTypeInv, varargout ] = formatVectors( InDims, OutDims, PermType, varargin )
%FORMATVECTORS Checks to see that the input arrays are in the format
% described by InDims. That is, the dimension specified by 'InDims'
% must have length 2 or 3. If not, 
% 
% INPUT ARGUMENTS
%   InDims:	[T/F, 1x1 sclar, 1D array, char]
%               InDims indicates which dimension is storing the vectors
%               ( default is dimension 1 )
%                   (1) true:	column storage (dimension 1)
%                      	false:	row storage (dimension 2)
%                   (2) Value indicates the dimension
%                   (3) Must have same length as number of inputs
%                       InDims(i) is the dimension for the i'th entry
%                   (4) 'cols': column storage (dimension 1)
%                       'rows':	row storage (dimension 2)
%               
%   OutDims:	[T/F, 1x1 sclar, 1D array, char]
%               OutDims forces the permutation indicated by InDims
%               for every input array without checking whether they already
%               satisfy the format or not
%               ( default is dimension 1 )
%                   Same format as 'InDims'
% 
%   PermType:	[T/F, 1x1 sclar, 1D array, char]
%               PermType indicates which type of permutation is made.
%               Generalized transpose swaps the dimensions indicated by
%               InDims(i) and OutDims(i), whereas shift-transpose shifts
%               OutDims(i) to the location of InDims(i) (equivalent in 2D)
%               ( default is generalized transpose )
%                   (1)	true:	generalized transpose
%                      	false:	shift-transpose
%                   (2)	1:	generalized transpose
%                      	2:	shift-transpose
%                   (3)	Must have same length as number of inputs
%                       InDims(i) is the permutation type (as in (2))
%                   (4)	'trans': generalized transpose
%                       'shift':	 shift-transpose
%               

%% Input Handling

% varargin
NumArrays	=   length( varargin );
varargout	=   cell( 1, NumArrays );

% InDims
DefaultIn	=   zeros( 1, NumArrays );
for ii = 1:NumArrays
    d	=   find(	( size(varargin{ii}) == 2 )	|	...
                    ( size(varargin{ii}) == 3 ),	...
                    true, 'first' );
    if isempty( d )
        %warning( 'No dimension has size 2 or 3 for argument #%d', ii );
        DefaultIn(ii)	=	NaN;
    else
        DefaultIn(ii)	=	d;
    end
end

InDims      =	parseInputArg( InDims, NumArrays, ...
                    [1,2], {'cols','rows'}, [1,2], DefaultIn );

% OutDims
OutDims     =	parseInputArg( OutDims, NumArrays, ...
                    [1,2], {'cols','rows'}, [1,2], [] );

% PermType
PermType	=	parseInputArg( PermType, NumArrays, ...
                    [1,2], {'trans','shift'}, [1,2], ones(1,NumArrays) );

%% Permute arrays
[ InDimsInv, OutDimsInv, PermTypeInv ] = deal( InDims, OutDims, PermType );

for ii = 1:NumArrays
    
    d	=   InDims(ii);
    D	=   OutDims(ii);
    
    if isnan(d)
        varargout{ii}	=	varargin{ii};
        continue;
    end
    
    arrSize     =   size( varargin{ii} );
    if D > length( arrSize )
        arrSize( end+1:D )	=   1;
    end
    
    arrOrder	=   1:length( arrSize );
    newArrOrder	=   arrOrder;
    
    if PermType(ii) == 1
        newArrOrder([d,D])	=	[D,d];
    else
        newArrOrder     =   circshift( newArrOrder, D - d , 2 );
        InDimsInv(ii)	=   OutDims(ii);
        OutDimsInv(ii)	=   InDims(ii);
    end
    
    varargout{ii}	=   permute( varargin{ii}, newArrOrder );
    
end

end

function InputArg	=   parseInputArg( InputArg, NumArrays, BoolVals, CharList, CharVals, DefaultValue )

argSize	=   [1, NumArrays];

if isempty( InputArg )
    InputArg	=   DefaultValue;
end

if      isa( InputArg, 'logical' )
    
    InputArg	=	repmat(	...
        BoolVals(1)*(InputArg) + BoolVals(2)*(~InputArg), argSize );
    
elseif	isa( InputArg, 'double' )
    
    if      isscalar( InputArg )
        InputArg	=   repmat( InputArg, argSize );
    elseif  isvector( InputArg )
        InputArg	=   reshape( InputArg, argSize );
    else
        warning( 'Input value is not a scalar/vector. Using default' );
        InputArg	=   DefaultValue;
    end
    
elseif	isa( InputArg, 'char' )
    
    found	=   false;
    
    for ii = 1:length(CharList)
        if strcmpi( InputArg, CharList{ii} )
            InputArg	=   CharVals(ii) * ones( argSize );
            found       =   true;
            break
        end
    end
    
    if ~found
        warning( 'Input argument %s is not valid. Using default value', ...
            InputArg );
        InputArg	=   DefaultValue;
    end
    
else
    InputArg	=   DefaultValue;
end


end
