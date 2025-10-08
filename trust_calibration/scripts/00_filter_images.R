{
  rm(list = ls())
  library(tidyverse)
  library(purrr)
  library(here)
  library(cowplot)
  library(jpeg)
  library(pdftools)
}

{
  run_main <- function(filter_data=FALSE, save_plots = FALSE) {
    paths <- list(
      data = here("melnet", "data", "predictions", "features_predictions.csv"),
      images = here("images", "resized"),
      figures = here("trust_calibration", "figures")
    )
    
    conf_lvl <- list(
      low = c(0, 30),
      med = c(35, 65),
      high= c(70, 100)
    )
    
    data <- get_data(paths$data)
    data <- set_conf_levels(data, conf_lvl)
    
    if (filter_data) {
      ## filter highly symptomatic images
      p = 0.6
      pctile <- list(
        p = p,
        sym = quantile(data$pi_sym, p, na.rm = TRUE),
        bor = quantile(data$pi_bor, p, na.rm = TRUE),
        col = quantile(data$pi_col, p, na.rm = TRUE)
      )
      rm(p)  
      data_filtered <- data %>%
        filter(pi_sym < pctile$sym & 
                 pi_bor < pctile$bor & 
                 pi_col < pctile$col &
                 confidence_category != "other")
    } else {
      data_filtered <- data
    }
      
    # images_per_level <- data_filtered %>%
    #   group_by(true_class, predicted_class, confidence_category) %>%
    #   summarise(n = n()) %>%
    #   ungroup()
  
    ## extract ids for specific condition ----
    true_class <- 0
    predicted_class <- 0
    conf_cats <- c("low", "med", "high")
    cases <- expand.grid(true_class = c(0, 1), predicted_class = c(0, 1), confidence_bin = conf_cats)
    
    all_ids <- pmap(cases, function(true_class, predicted_class, confidence_bin) {
      extract_ids(data_filtered, true_class, predicted_class, confidence_bin)
    })
    names(all_ids) <- apply(cases, 1, function(row) paste(row, collapse = "_"))
  
  
    if (save_plots) {
      ### plot/check identified images ----
      for (lvl in seq_along(all_ids)) {
        fname <- paste0(names(all_ids)[[lvl]], ".pdf")
        plot_images(all_ids[[lvl]], paths$images, max_images = 10)
        w <- dev.size("in")[1] * 2
        h <- dev.size("in")[2] * 2
        ggsave(here(paths$figures, fname), width = w, height = h, device="pdf")
      }
      
      ## combine
      all_files <- list.files(here(paths$figures), pattern = "*.pdf", full.names = TRUE)
      pdf_files <- all_files[!basename(all_files) %in% "all_confidence_levels.pdf"]
      combined_pdf <- pdf_combine(pdf_files, output = here(paths$figures, "all_confidence_levels.pdf"))
    }
    
    if (filter_data) {
      return(list(data=data, all_ids=all_ids))
    } else {
      return(list(data=data, data_filtered=data_filtered, all_ids=all_ids))
    }
  } 
  
  get_data <- function(path) {
    data <- read.csv(path) %>%
      mutate(
        confidence = round(abs(0.5 - pred_best) * 200),
      ) %>%
      mutate(across(starts_with("pred"),
                    list(class = ~ ifelse(. > 0.5, 1, 0)),
                    .names = "pred_class_{col}")) %>%
      rename_with(~ gsub("pred_class_pred_", "pred_class_", .), starts_with("pred_class_")) %>%
      select(id, true_class, starts_with("pred"), confidence, everything()) %>%
      mutate(confidence_bin = cut(confidence, breaks = seq(0, 100, by=5), include.lowest=T))
    return(data)
  }
  
  
  set_conf_levels <- function(data, conf_lvl) {
    data <- data %>%
      mutate(confidence_category = case_when(
        confidence >= conf_lvl$low[1] & confidence <= conf_lvl$low[2] ~ "low",
        confidence >= conf_lvl$med[1] & confidence <= conf_lvl$med[2] ~ "med",
        confidence >= conf_lvl$high[1] & confidence <= conf_lvl$high[2] ~ "high",
        TRUE ~ "other",
      ))
    return(data)
  }
  
  
  extract_ids <- function(df, true_, pred_, conf_) {
    ids <- df %>%
      filter(true_class == true_ & 
               predicted_class == pred_ & 
               confidence_category == conf_) %>%
      select(id, true_class, predicted_class, confidence, starts_with("pred"), confidence_category)
    return(ids)
  }
  
  plot_images <- function(ids, image_dir, max_images) {
    plots <- list()
    n_images <- length(ids$id)
    
    max_images <- min(max_images, n_images)
    ids <- ids[1:max_images, ]
    
    for (i in seq_along(ids$id)) {
      img_path <- file.path(image_dir, paste0(ids$id[i],  ".jpg"))
      img <- readJPEG(img_path)
      
      p <- ggplot() + 
        annotation_raster(img, xmin=-Inf, xmax=Inf, ymin= -Inf, ymax = Inf) + 
        theme_void() + 
        ggtitle(paste0(ids$id[i], "\nConf: ", ids$confidence[i],"%"))
      plots[[i]] <- p
    }
    num_images <- length(plots)
    n_cols <- 5
    n_rows <- ceiling(num_images / n_cols)
    if (num_images > max_images) {
      plots <- plots[1:max_images]
      n_rows <- ceiling(max_images / n_cols)
    }
    plot_grid_base <- plot_grid(plotlist = plots, ncol = n_cols, nrow=n_rows)
    
    true_class <- ids$true_class[1]
    predicted  <- ids$predicted_class[1]
    conf_level <- ids$confidence_category[1]
    title <- ggdraw() + 
      draw_label(paste0(
        "Class: ", true_class,
        ", Predicted: ", predicted,
        ", Conf Level: ", conf_level,
        "\ntotal images: ", n_images),
                 fontface='bold', x=0.5, hjust=0.5) 
    
    final_plot <- plot_grid(title, plot_grid_base, ncol=1, rel_heights=c(0.1, 1))
    return(final_plot)
  }
}


{
  save_plots = FALSE
  filter_data = FALSE
  result <- run_main(filter_data, save_plots)
  
  data <- result$data
  if (filter_data) {
    data_filtered <- result$data_filtered
  }
  all_ids <- result$all_ids
  
  
  {
    ambigu <- list(ous=0.05) #0.167)
    ambigu$min <- 0.5 - ambigu$ous
    ambigu$max <- 0.5 + ambigu$ous
    
    #pred_716, pred_759, pred_816, pred_best
    d_all <- data %>%
      select(-starts_with("pred_class"), -pred_816) %>%
      filter(if_all(starts_with("pred"), ~ . <= 0.1))
      # filter(if_all(starts_with("pred"), ~ . >= ambigu$min & . <= ambigu$max))
  
    d_any <- data %>%
      select(-starts_with("pred_class"), -pred_816) %>%
      filter(if_any(starts_with("pred"), ~ . > 0.9))
      # filter(if_any(starts_with("pred"), ~ . >= ambigu$min & . <= ambigu$max))
  }
  
}

