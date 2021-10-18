#####
# 
# Notation:
# 
# Y: input data
# 
# θ: current pseudo label
# 
# θ′: proposed label from CVAE posterior
# 
# E1: Y -> (μr, logσr)
#   Encoder mapping input data Y to a multivariate Gaussian N(μr, logσr),
#   which parameterizes a latent space.
# 
# E2: (Y, θ) -> (μq, logσq)
#   Encoder mapping input data Y and labels θ and to a multivariate Gaussian N(μq, logσq),
#   which parameterizes a latent space.
# 
# D: (Y, z) -> (μθ, logσθ)
#   Decoder mapping input data Y and latent space sample z, which will be either zr ~ N(μr, logσr)
#   or zq ~ N(μq, logσq), to a multivariate Gaussian N(μθ, logσθ) which parameterizes
#   the posterior distribution of θ.
# 
####

function train_metropolis_hastings_cvae(Y, θ, E1, E2, D)
    # Sample proposed pseudo labels
    θ′ = sample_cvae_posterior(Y, E1, D)

    # Update pseudo labels using Metropolis-Hastings update rule
    θ  = metropolis_hastings_update(Y, θ, θ′)

    # Perform gradient descent step on E1, E2, and D parameters using the
    # cross-entropy loss with Y and the updated pseudo labels θ
    gradient_descent_step() do
        ℓ = cross_entropy_loss(Y, θ, E1, E2, D)
    end
end

function cross_entropy_loss(Y, θ, E1, E2, D)
    # Compute the latent spaces
    μr, logσr = E1(Y)
    μq, logσq = E2(Y, θ)

    # Compute the KL-divergence between the latent spaces (μq, logσq) and (μr, logσr)
    KL = (exp(2 * (logσq - logσr)) + exp(-2 * logσr) * (μr - μq)^2 - 1) / 2 - (logσq - logσr)

    # Sample from (μq, logσq), decode the latent space sample, and compute the evidence lower bound
    zq        = μq + randn() * exp(logσq)
    μθ, logσθ = D(Y, zq)
    ELBO      = ((exp(-logσθ) * (θ - μθ))^2 + log2π) / 2 + logσθ

    # Return cross-entropy lower bound loss
    ℓ = KL + ELBO

    return ℓ
end

function sample_cvae_posterior(Y, E1, D)
    # Compute the latent space and sample from it
    μr, logσr = E1(Y)
    zr        = μr + randn() * exp(logσr)

    # Decode the latent space sample and sample from the posterior
    μθ, logσθ = D(Y, zr)
    θ         = μθ + randn() * exp(logσθ)

    return θ
end

function metropolis_hastings_update(Y, θ, θ′)
    # Compute the log-likelihood probability of Y under θ and θ′
    logPYθ  = loglikelihood(Y, θ)
    logPYθ′ = loglikelihood(Y, θ′)

    # Compute the Metropolis-Hastings acceptance ratio, and sample
    # u ~ Uniform(0,1). Accept θ′ if α > u
    α = min(1, exp(logPYθ′ - logPYθ))
    u = rand()

    return α > u ? θ′ : θ
end

function loglikelihood(Y, θ)
    # Example simple Gaussian log-likelihood:
    #   Y: homodyne filtered input image
    #   θ: inverse-filtered image X and uniform noise level logϵ,
    #      i.e. θ = [X; logϵ] where X ~ homodyne⁻¹(Y)

    # Unpack parameters, compute forward homodyne filter, and compute Gaussian log-likelihood.
    # Ignore additive log2π/2 constant as only the difference of log-likelihoods,
    # i.e. the ratio of likelihoods, is relevent.
    X, logϵ = θ
    Ŷ       = homodyne(X)
    logPYθ  = -1/2 * exp(-2 * logϵ) * sum((Y - Ŷ)^2)

    return logPYθ
end
