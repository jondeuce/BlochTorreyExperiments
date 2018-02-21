classdef GaussianKernel
    %GAUSSIANKERNEL Summary of this class goes here
    %   Detailed explanation goes here
    
    properties ( GetAccess = public, SetAccess = immutable )
        sigma % standard deviation (unitfull)
        gsize % grid size (unitless)
        gdims % grid dimensions (unitfull)
        vsize % size of subvoxels (i.e. gdims./gsize)
    end
    
    properties ( GetAccess = private )
        dim % dimension of convolution
        K   % precomputed Kernel in the fourier domain
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % CLASS CONSTRUCTOR:
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods
        
        function G = GaussianKernel(sig, gsiz, gdim)
            
            % Default to unit-length isotropic subvoxels
            if nargin < 3
                gdim = gsiz;
            end
            
            % Force all inputs to be 1xn vectors
            sig  = sig(:).';
            gsiz = gsiz(:).';
            gdim = gdim(:).';
            
            if length(gsiz) ~= length(gdim) || ~any(length(gsiz) == [2,3])
                error('gdims and gsize must both be 2- or 3-element vectors');
            end
            
            if isscalar(sig)
                sig = sig * ones(1,length(gsiz));
            end
            
            if length(sig) ~= length(gsiz)
                error('sigma must be a scalar, or the same length as gsize and gdims');
            end
            
            % Set sigma to zero for singleton dimensions
            sig(gsiz==1) = 0;
            
            % Dimension is simplify length of size vector; 1D and 2D can be
            % handled together as a 2D kernel with one singleton dimension
            ndim = length(gsiz);
            if ndim > 3
                error('Only supports up to 3 dimensions');
            end
            
            % Assign public class properties
            G.sigma = sig;
            G.gsize = gsiz;
            G.gdims = gdim;
            G.vsize = gdim./gsiz;
            G.dim   = ndim;
            
            % Assign private properties
            G.K = [];
            
        end
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % PUBLIC METHODS:
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods ( Access = public, Static = false )
        
        function G = precompute(G)
            
            if all(G.sigma == 0)
                G.K = 1;
                return
            end
            
            switch G.dim
                case 2
                    g2D = Geometry.GaussianKernel.unitGaussian2D(G.vsize, G.sigma, G.gsize);
                    G.K = Geometry.GaussianKernel.padfastfft( g2D, G.gsize - size(g2D), true, 0 );
                case 3
                    g3D = Geometry.GaussianKernel.unitGaussian3D(G.vsize, G.sigma, G.gsize);
                    g3Dsize = size(g3D); if numel(g3Dsize) == 2; g3Dsize(3) = 1; end
                    G.K = Geometry.GaussianKernel.padfastfft( g3D, G.gsize - g3Dsize, true, 0 );
                otherwise
                    error('Only 2D and 3D implemented');
            end
            
            G.K = fftn( ifftshift( G.K ) );
            
            % Want to allow for complex diffusion coefficients, where
            % sigma = sqrt(2*D*t) along each dimension, with each D 
            % possibly complex
            if all(isreal(G.sigma))
                G.K = real(G.K);
            end
            
        end
        
        function y = conv(G,x)
            
            if ~isa(G,'Geometry.GaussianKernel')
                y = conv(x,G);
                return
            end
            
            if ~isPrecomputed(G)
                G = precompute(G);
            end
            
            if all(G.sigma == 0) || isequal(G.K,1)
                y = x;
            else
                y = ifftn(fftn(x).*G.K);
            end
            
        end
                
        function g = kernel(G)
            switch G.dim
                case 2
                    g = Geometry.GaussianKernel.unitGaussian2D(G.vsize, G.sigma, G.gsize);
                case 3
                    g = Geometry.GaussianKernel.unitGaussian3D(G.vsize, G.sigma, G.gsize);
                otherwise
                    error('Only 2D and 3D implemented');
            end
        end
        
        function b = isPrecomputed(G)
            b = ~isempty(G.K);
        end
                
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % PUBLIC STATIC METHODS:
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods ( Access = public, Static = true )
        
        % unitGaussian1D: generates 1D kernel along 'dim' for a given vox
        % and (possibly complex-valued) sig 
        [g,I] = unitGaussian1D(vox, sig, dim)
        
        % padfastfft: pads kernel x to desired size
        [x] = padfastfft(x, padSize, forceSize, method)
        
        function g = gaussianClamp1D(g, maxlen)
            % Want to make sure that if the kernel length is too large that
            % the center value of the kernel remains at the center (for
            % maxlen odd) and one to the right of center (for maxlen even),
            % as is consistent with FFT.
            if length(g) > maxlen
                warning('Kernel size is larger than input array; cropping kernel');
                relidx = -floor(maxlen/2):(ceil(maxlen/2)-1);
                idx = floor(length(g)/2) + 1 + relidx;
                g = g(idx);
            end
        end
        
        function [g2D,I2D] = unitGaussian2D(vox, sig, maxSize)
            [g1,I1] = Geometry.GaussianKernel.unitGaussian1D(vox(1), sig(1), 1);
            [g2,I2] = Geometry.GaussianKernel.unitGaussian1D(vox(2), sig(2), 2);
            
            g1 = Geometry.GaussianKernel.gaussianClamp1D(g1, maxSize(1));
            g2 = Geometry.GaussianKernel.gaussianClamp1D(g2, maxSize(2));
            
            % 2D kernel is just product of 1D kernels along each dimension;
            % renormalizing is just to be safe
            g2D = g1 * g2;
            I2D = I1 * I2;
            g2D = g2D * (I2D/sum(g2D(:)));
        end
        
        function g3D = unitGaussian3D(vox, sig, maxSize)
            [g2D,I2D] = Geometry.GaussianKernel.unitGaussian2D(vox(1:2), sig(1:2), maxSize(1:2));
            
            [g3,I3] = Geometry.GaussianKernel.unitGaussian1D(vox(3), sig(3), 3);
            g3 = Geometry.GaussianKernel.gaussianClamp1D(g3, maxSize(3));
            
            % 3D kernel is just product of 2D kernel in dims = 1,2 and 1D
            % kernel along dim = 3; renormalizing is just to be safe
            g3D = bsxfun(@times, g2D, g3);
            I3D = I2D * I3;
            g3D = g3D * (I3D/sum(g3D(:)));
        end
        
    end
    
end
