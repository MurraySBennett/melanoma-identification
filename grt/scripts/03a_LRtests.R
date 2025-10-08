## load libraries ----
{
  rm(list = ls())
  library(tidyverse)
  library(here)
  library(grtools)
  
  min_acc <- 0.25
  data <- read.csv(here("grt", "data", "final", paste0("mel_grt_data_acc_", min_acc, ".csv")))
  
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
  
  cm_ab <- get_cm(data, "ab")
  cm_ac <- get_cm(data, "ac")
  cm_bc <- get_cm(data, "bc")

  our_cms <- tail(cm_ab, 2)
  cm_ab <- head(cm_ab, -2)


  # read models ----
  our_models <- readRDS(here("grt", "model_outputs", "JWH_MSB_models.rds"))

  ab_models <- readRDS(here("grt", "model_outputs", "ab_models.rds"))
  ac_models <- readRDS(here("grt", "model_outputs", "ac_models.rds"))
  bc_models <- readRDS(here("grt", "model_outputs", "bc_models.rds"))

  wind_ab <- readRDS(here("grt", "model_outputs", paste0("ab_model_wind_acc_", min_acc, ".rds")))
  wind_ac <- readRDS(here("grt", "model_outputs", paste0("ac_model_wind_acc_", min_acc, ".rds")))
  wind_bc <- readRDS(here("grt", "model_outputs", paste0("bc_model_wind_acc_", min_acc, ".rds")))
}

# {
#   ab_wald <- wind_ab
#   ab_wald$model <- "GRT-wIND_full"
#   ab_wald <- wald(ab_wald, cm_ab, estimate_hess=T)
#   
#   ac_wald <- wind_ac
#   ac_wald$model <- "GRT-wIND_full"
#   ac_wald <- wald(ac_wald, cm_ac, estimate_hess=T)
#   
#   bc_wald <- wind_bc
#   bc_wald$model <- "GRT-wIND_full"
#   bc_wald <- wald(bc_wald, cm_bc, estimate_hess=T)
# }

{
  lr_test_list <- c("PS(A)", "PS(B)", "PI")
  print("==== AB Likelihood ratio tests =====")
  ab_lr <- lr_test(wind_ab, cm_ab, test=lr_test_list)
  summary(ab_lr)
  
  print("==== AC Likelihood ratio tests =====")
  ac_lr <- lr_test(wind_ac, cm_ac, test=lr_test_list)
  summary(ac_lr)
  
  print("==== BC Likelihood ratio tests =====")
  bc_lr <- lr_test(wind_bc, cm_bc, test=lr_test_list)
  summary(bc_lr)
}

