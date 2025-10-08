{
  rm(list=ls())
  library(here)
  library(tidyverse)
  library(devtools)
  
  # an updated grtools, installed via:
  # devtools::install_github("MurraySBennett/grtools_abcNoise")
  # library(grtools)
  
  # if uninterested in continuously installing -- run devtools::load_all() from the grtools_abcNoise project
  load_all()
  
  data <- read.csv(here("..", "melanoma-identification", "grt", "data", "final", "mel_grt_data.csv"))
  cms <- data %>%
    filter(response != '') %>%
    group_by(pID) %>%
    summarize(
      cm = list(table(stimulus, response))
    ) %>%
    deframe()
  
  cmat <- as.matrix(cms[[1]])
}

{
  # fit and plot all
  hm_fits <- list()
  counter <- 1
  alpha <- 0.3
  noise_models = list("none", "uniform") # c("differential", "2")
  
  marginals=F
  scatter=F
  show_assumptions=T
  show_labels=T 
  ellipse_width=0.8
  labels=c("Dim1", "Dim2")
  
  for (cm in cms) {
    result <- grt_hm_fit(cm, noise_models=noise_models, alpha=alpha)
    for (model_name in names(result)){
      model <- result[[model_name]]
      hm_fits[[model_name]][[counter]] <- model
      subtitle <- paste(model_name, "| alpha:", alpha)
      plot(
        model, labels=labels,
        marginals=marginals, scatter=scatter,
        show_assumptions=show_assumptions, show_labels=show_labels,
        subtitle=subtitle, ellipse_width=ellipse_width
      )
    }
    counter <- counter + 1
  }
}


## https://www.semanticscholar.org/reader/715b69575dadd7804b4f8ccb419a3ad8b7b7ca89
{
  print(cmat)
  # Test with alpha = 0 (no noise influence)
  hm_fit_results_alpha0 <- grt_hm_fit(cmat, alpha = 0)
  summary(hm_fit_results_alpha0)
  
  # Test with alpha = 0.5 (equal influence of GRT and noise) and uniform noise
  hm_fit_results_alpha5_uniform <- grt_hm_fit(cmat, alpha = 0.5, noise_models = list(c("uniform")))
  summary(hm_fit_results_alpha5_uniform)
  
  # Test with alpha = 0.5 and differential noise (noise_ratio = 2)
  hm_fit_results_alpha5_differential <- grt_hm_fit(cmat, alpha = 0.5, noise_models = list(c("differential", 2)))
  summary(hm_fit_results_alpha5_differential)
  
  # Test with alpha = 1 (noise only) and uniform noise
  hm_fit_results_alpha1_uniform <- grt_hm_fit(cmat, alpha = 1, noise_models = list(c("uniform")))
  summary(hm_fit_results_alpha1_uniform)
  
  # Test with alpha = 1 and differential noise (noise_ratio = 2)
  hm_fit_results_alpha1_differential <- grt_hm_fit(cmat, alpha = 1, noise_models = list(c("differential", 2)))
  summary(hm_fit_results_alpha1_differential)
  
  # test all noise models and a variety of alpha values.
  hm_fit_results_all <- grt_hm_fit(cmat, alpha = .25)
  summary(hm_fit_results_all)
  
  hm_fit_results_all <- grt_hm_fit(cmat, alpha = .75)
  summary(hm_fit_results_all)
}



{
  plot(hm_fit_results_alpha0, c("Dim 1", "Dim 2"),
       marginals = FALSE, scatter = FALSE,
       show_assumptions = TRUE, show_labels = TRUE, subtitle = "Alpha = 0",
       ellipse_width = 0.8
  )
  
  plot(hm_fit_results_alpha5_uniform$`GRT-uniform`$best_model, c("Dim 1", "Dim2"),
       marginals = FALSE, scatter = FALSE,
       show_assumptions = TRUE, show_labels = TRUE, subtitle = "Alpha = 0.5, Uniform Noise",
       ellipse_width=0.8)
  
  plot(hm_fit_results_alpha5_differential$`GRT-differential-2`$best_model, c("Dim 1", "Dim2"),
       marginals = FALSE, scatter = FALSE,
       show_assumptions = TRUE, show_labels = TRUE, subtitle = "Alpha = 0.5, Differential Noise",
       ellipse_width=0.8)
}

