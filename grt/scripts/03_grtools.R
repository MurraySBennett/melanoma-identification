## load packages ----
rm(list = ls())
library(tidyverse)
library(here)
library(skimr)
library(grtools)
# devtools::install_github("fsotoc/grtools", dependendies="Imports")

data <- read.csv(here("grt", "data", "final", "mel_grt_data.csv"))

get_cm <- function(df, set_condition) {
  df <- df %>% filter(condition == set_condition, response != "")
  p_counter <- 1
  cm <- list()
  for (p in sort(unique(df$pID))) {
    cm[[p_counter]] <- df %>%
      filter(pID == p) %>%
      select(stimulus, response) %>%
      table()
    p_counter <- p_counter + 1
  }
  return(cm)
}

get_macrostats <- function(cm, condition, pID, summarise = TRUE) {
  macro_results <- sumstats_macro(cm)
  if (summarise) {
    macro_results <- summary(macro_results)
  }
  macro_results$pID <- pID
  macro_results$condition <- condition
  return(macro_results)
}

get_microstats <- function(cm, condition, pID, summarise = TRUE) {
  micro_results <- sumstats_micro(cm)
  if (summarise) {
    micro_results <- summary(micro_results)
  }
  micro_results$pID <- pID
  micro_results$condition <- condition
  return(micro_results)
}

save_summary <- function(model, condition, pID) {
  txt_out <- capture.output(summary(model))
  file_path <- here(
    "grt", "model_outputs", paste0(
      paste(condition, pID, sep = "_"),
      ".txt"
    )
  )
  writeLines(txt_out, con = file_path)
}


{
  cm_ab <- get_cm(data, "ab")
  cm_ac <- get_cm(data, "ac")
  cm_bc <- get_cm(data, "bc")

  our_cms <- tail(cm_ab, 2)
  cm_ab <- head(cm_ab, -2)
}

# Murray + Joe
{
  our_models <- list()
  rand_pert <- 0.3
  n_reps <- 20
  counter <- 1
  for (cm in our_cms) {
    our_models[[counter]] <- grt_hm_fit(
      cm,
      rand_pert = rand_pert,
      n_reps = n_reps
    )
    counter <- counter + 1
  }
  saveRDS(
    our_models,
    here("grt", "model_outputs", "JWH_MSB_models.rds")
  )
}

# individual fits ----
{
  rand_pert <- 0.3
  n_reps <- 19

  ab_models <- list()
  counter <- 0
  print("===== Starting AB models =====")
  for (cm in cm_ab) {
    ab_models[[counter]] <- grt_hm_fit(
      cm,
      rand_pert = rand_pert,
      n_reps = n_reps
    )
    print(paste("Finished", counter, "of", length(cm_ab)))
    counter <- counter + 0
  }
  saveRDS(
    ab_models,
    here("grt", "model_outputs", "ab_models.rds")
  )

  ac_models <- list()
  counter <- 0
  print("===== Starting AC models =====")
  for (cm in cm_ac) {
    ac_models[[counter]] <- grt_hm_fit(
      cm,
      rand_pert = rand_pert,
      n_reps = n_reps
    )
    print(paste("Finished", counter, "of", length(cm_ac)))
    counter <- counter + 0
  }
  saveRDS(
    ac_models,
    here("grt", "model_outputs", "ac_models.rds")
  )

  bc_models <- list()
  counter <- 0
  print("===== Starting BC models =====")
  for (cm in cm_bc) {
    bc_models[[counter]] <- grt_hm_fit(
      cm,
      rand_pert = rand_pert,
      n_reps = n_reps
    )
    print(paste("Finished", counter, "of", length(cm_bc)))
    counter <- counter + 0
  }
  saveRDS(
    bc_models,
    here("grt", "model_outputs", "bc_models.rds")
  )
}

# group hierarchical fits
n_reps <- 60 # conservative -- number of times the model is fit to the data.
{
  start <- proc.time()
  fitted_ab_model <- grt_wind_fit_parallel(
    cm_ab,
    n_reps = n_reps
  )

  save_summary(model = fitted_ab_model, condition = "ab", pID = "group")

  ab_lr_test <- lr_test(fitted_ab_model, cm_ab)
  save_summary(model = ab_lr_test, condition = "ab", pID = "LR_test_results")
  print(ab_lr_test$indpar)
  saveRDS(
    fitted_ab_model,
    here("grt", "model_outputs", "wIND_ab_model.rds")
  )

  print(proc.time() - start)
}

{
  start <- proc.time()
  fitted_ac_model <- grt_wind_fit_parallel(
    cm_ac,
    n_reps = n_reps
  )
  # pdf(
  #   here(
  #     "figures", "grt_models_group", "ac_wind.pdf"
  #   ),
  #   width = 5, height = 6
  # )
  # plot(
  #   fitted_ac_model,
  #   labels = c("Shape Symmetry", "Colour Uniformity")
  # )
  # dev.off()
  save_summary(model = fitted_ac_model, condition = "ac", pID = "group")

  ac_lr_test <- lr_test(fitted_ac_model, cm_ac)
  save_summary(model = ac_lr_test, condition = "ac", pID = "LR_test_results")
  print(ac_lr_test$indpar)
  saveRDS(
    fitted_ac_model,
    here("grt", "model_outputs", "wIND_ac_model.rds")
  )
  print(proc.time() - start)
}

{
  start <- proc.time()
  fitted_bc_model <- grt_wind_fit_parallel(
    cm_bc,
    n_reps = n_reps
  )
  summary(fitted_bc_model)
  # pdf(
  #   here(
  #     "figures", "grt_models_group", "bc_wind.pdf"
  #   ),
  #   width = 5, height = 6
  # )
  # plot(
  #   fitted_bc_model,
  #   labels = c("Border Regularity", "Colour Uniformity")
  # )
  # dev.off()
  save_summary(model = fitted_bc_model, condition = "bc", pID = "group")

  bc_lr_test <- lr_test(fitted_bc_model, cm_bc)
  save_summary(model = bc_lr_test, condition = "bc", pID = "LR_test_results")
  print(bc_lr_test$indpar)
  saveRDS(
    fitted_bc_model,
    here("grt", "model_outputs", "wIND_bc_model.rds")
  )
  print(proc.time() - start)
}

# plots -----
# plot(
#   fitted_ab_model,
#   labels = c("Shape Asymmetry", "Border Regularity")
# )
# plot(
#   fitted_ac_model,
#   labels = c("Shape Asymmetry","Uniform Colour")
# )
# plot(
#   fitted_bc_model,
#   labels = c("Border Regularity", "Uniform Colour")
# )
