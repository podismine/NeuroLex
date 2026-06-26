#!/usr/bin/env Rscript

# Remove participant-level training data from serialized GAMLSS models while
# retaining their fitted parameters, formulae, smoothing knots, and site-level
# random effects. Run from any directory:
#   Rscript sanitize_rds_models.R [model_directory]

args <- commandArgs(trailingOnly = TRUE)
model_dir <- if (length(args)) args[[1]] else "."
paths <- list.files(model_dir, pattern = "\\.rds$", full.names = TRUE,
                    ignore.case = TRUE)

strip_observation_data <- function(x, n_observations) {
  # Formulae and functions carry model definitions, not training rows.
  if (inherits(x, c("formula", "terms", "call", "name", "function"))) {
    return(x)
  }

  # Participant-level vectors, matrices, arrays, and data frames are removed.
  if (is.atomic(x) && !is.null(x) && length(x) == n_observations) {
    return(NULL)
  }
  if (is.matrix(x) || is.data.frame(x) || is.array(x)) {
    dimensions <- dim(x)
    if (length(dimensions) && dimensions[[1]] == n_observations) {
      return(NULL)
    }
    return(x)
  }

  if (!is.list(x)) return(x)

  cleaned <- x
  for (element in names(x)) {
    cleaned[[element]] <- strip_observation_data(x[[element]], n_observations)
  }
  cleaned
}

for (path in paths) {
  model <- readRDS(path)
  if (!inherits(model, "gamlss") || is.null(model$N)) {
    stop(sprintf("%s is not a GAMLSS model with a sample count", path))
  }

  cleaned <- strip_observation_data(model, model$N)
  class(cleaned) <- class(model)

  temporary_path <- paste0(path, ".tmp")
  saveRDS(cleaned, temporary_path)
  if (!file.rename(temporary_path, path)) {
    unlink(temporary_path)
    stop(sprintf("Could not replace %s", path))
  }
  message(sprintf("Sanitized %s", basename(path)))
}
