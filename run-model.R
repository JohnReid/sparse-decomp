#!/usr/bin/Rscript

args <- commandArgs(trailingOnly = TRUE)
stopifnot(2 == length(args))
input.data <- args[1]
output.file <- args[2]

# Load data
message('**** Loading data from: ', input.data)
load(input.data)
stopifnot(exists("stan.data"))

# Source config file if we have one
config.file <- "run-model-cfg.R"
if (file.exists(config.file)) {
    message('**** Sourcing config from: ', config.file)
    source(config.file)
}

# Load libraries
library(rstan)
library(parallel)
library(dplyr)
library(reshape2)

# Compile and fit model in parallel
num.cores <- getOption("stan.num.cores", detectCores() - 1)
chains <- getOption("stan.chains", num.cores)
iter <- getOption("stan.iter", 1000)
thin <- getOption("stan.thin", 10)
model.file <- 'sparse-decomp.stan'
compiled.model.file <- 'spares-decomp-model.rds'
if (file.exists(compiled.model.file)) {
    message("**** Loading pre-compiled model from ", compiled.model.file)
    compiled <- readRDS(compiled.model.file)
} else {
    message("**** Compiling model")
    compiled <- stan(file=model.file, chains=0)
    message("**** Saving compiled model to ", compiled.model.file)
    saveRDS(compiled, compiled.model.file)
}
sflist <- mclapply(1:num.cores,
                   mc.cores=num.cores,
                   function(i)
                       stan(fit=compiled,
                            model_name='sparse-decomp',
                            data=stan.data,
                            thin=thin,
                            iter=iter,
                            seed=i,
                            chains=chains,
                            chain_id=i,
                            refresh=-1))
fit <- sflist2stanfit(sflist)
rm(sflist, compiled)

# Summarise posterior
posterior.summary <- monitor(fit, print=FALSE)

# Function to extract samples from fit
melt.samples <- function(sample.list, sample.dims) {
    melt.var <- function(var.name) {
        melt(sample.list[[var.name]],
            c("iter", sample.dims[[var.name]]),
            value.name=var.name)
    }
    sapply(names(sample.dims), melt.var)
}

# Extract samples
sample.dims <- list(
    lp__ = c(),
    # Y = c("r", "t"),
    X = c("g", "t"),
    beta = c("g", "r"),
    s = c("g", "r"))
samples <- melt.samples(extract(fit, permuted=TRUE), sample.dims)
best.sample <- (samples$lp__ %>% arrange(-lp__) %>% head(1))$iter

# Save results
message('**** Saving output to: ', output.file)
save(posterior.summary,
     sample.dims,
     samples,
     best.sample,
     file=output.file)
