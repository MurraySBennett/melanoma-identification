# combine predictions with feature estimates
{
  rm(list = ls())
  library(tidyverse)
  library(here)

  get_data <- function(predictions_path, features_path, split_path) {
    preds <- read.csv(predictions_path) %>%
      rename_with(~ gsub("probability", "pred", .), starts_with("probability")) %>%
      rename_with(~ gsub("pred_acc_", "pred_", .), starts_with("pred_acc_")) %>%
      rename(true_class = malignant)

    feats <- read.csv(features_path) %>%
      mutate(cv_sym = round(x_sym + y_sym / 2, 3)) %>%
      select(id, cv_sym, cv_bor = compact, cv_col = rms, pi_sym, pi_bor, pi_col) %>%
      mutate_at(vars(cv_bor, cv_col, pi_sym, pi_bor, pi_col), round, 3)
    
    splits <- read.csv(split_path)
    splits$id <- gsub("\\.JPG$", "", splits$id)
    
    data <- preds %>% 
      inner_join(feats, by="id") %>%
      inner_join(splits, by="id") %>%
      select(id, true_class, everything())
  return(data)
  }
}

{
  paths <- list(
    predictions = here("melnet", "data", "predictions", "all_model_predictions.csv"),
    features = here("pwc", "data", "estimates", "btl_cv_data_revised.csv"),
    splits = here("melnet", "data", "image_splits.csv"),
    images = here("images", "resized"),
    figures = here("trust_calibration", "figures"),
    data = list(
      melnet = here("melnet", "data", "predictions"),
      pwc = here("pwc", "data", "estimates"),
      trust = here("trust_calibration", "data")
    )
  )
    
  data <- get_data(paths$predictions, paths$features, paths$splits)
  
  for (p in seq_along(paths$data)) {
    write.csv(data, here(paths$data[[p]], "features_predictions.csv"), row.names = FALSE)
  }
  rm(p)
}
