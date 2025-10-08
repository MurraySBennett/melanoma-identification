{
  rm(list=ls())
  library(tidyverse)
  library(here)
  library(readr)
  
  
}


{
  process_file <- function(file_path) {
    data <- tryCatch({
      read_csv(file_path, show_col_types=FALSE)
    }, error = function(e) {
      return(NULL)
    })
    if (is.null(data) | nrow(data) < 10) {
      return(NULL)
    }
    
    cond <- data$condition[2]
    image_conditions <- c(
      "ISIC_001135.JPG" = "sym",
      "ISIC_0056991.JPG" = "sym",
      "ISIC_1264950.JPG" = "sym",
      "ISIC_9972518.JPG" = "sym",
      "ISIC_0030831.JPG" = "bor",
      "ISIC_1805163.JPG" = "bor",
      "ISIC_0071367.JPG" = "col"
    )
    
    data <- data %>%
      mutate(
        path = file_path,
        condition = case_when(
          grepl("symmetry", cond) ~ "sym",
          grepl("regular", cond) ~ "bor",
          grepl("f", cond) ~ "col",
          TRUE ~ "NA"
        ),
        target_left = case_when(
          condition == "sym" & img_left %in% names(image_conditions)[image_conditions == "sym"] ~ img_left,
          condition == "bor" & img_left %in% names(image_conditions)[image_conditions == "bor"] ~ img_left,
          condition == "col" & img_left %in% names(image_conditions)[image_conditions == "col"] ~ img_left,
          TRUE ~ NA_character_
        ),
        target_right = case_when(
          condition == "sym" & img_right %in% names(image_conditions)[image_conditions == "sym"] ~ img_right,
          condition == "bor" & img_right %in% names(image_conditions)[image_conditions == "bor"] ~ img_right,
          condition == "col" & img_right %in% names(image_conditions)[image_conditions == "col"] ~ img_right,
          TRUE ~ NA_character_
        )
      ) %>%
      filter(sender=='trial', duration > 1000, duration < 10000) %>%
      select(img_left, img_right, condition, path, target_left, target_right) %>%
      filter(!is.na(target_left) | !is.na(target_right))
    if (nrow(data) == 0) {
      return(data.frame(
        img_left = character(),
        img_right = character(),
        condition = character(),
        path = character(),
        target_left = character(),
        target_right = character()
      ))
    }
    return(data)
  }
}


{
  files <- list.files(here("pwc", "data", "raw", "offline"), pattern = "\\.csv$", full.names = TRUE)
  # files <- list.files(here("pwc", "data", "raw"), pattern = "\\.csv$", full.names = TRUE)
  
  # f <- files[1]
  # df <- process_file(f)
  
  data <- data.frame()
  for (file_path in files) {
      processed_data <- process_file(file_path)
      data <- bind_rows(data, processed_data)
  }
  
  targets <- c(unique(data$target_left), unique(data$target_right))
  print(unique(targets))
  files_to_check <- unique(data$path)
}

