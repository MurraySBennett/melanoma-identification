# which sections to run
initialise <- TRUE
if (initialise) {
  rm(list=ls())
  library(tidyverse)
  library(here)
  library(janitor)
  library(jsonlite)
  library(readr)

  read_data <- TRUE
  save_raw <- TRUE
  save_processed <- TRUE
}


{
  process_file <- function(file_path, file_counter) {
    data <- tryCatch({
      read_csv(file_path, show_col_types=FALSE)
    }, error = function(e) {
      return(NULL)
    })
    if (is.null(data) | nrow(data) < 10) {
      print(paste("Skipping", file_path, ". Too small."))
      return(NULL)
    }
    url <- tryCatch({
      fromJSON(unique(data$url)[1])
    }, error = function(e) {
      print(paste("Skipping", file_path, "because of", e$message))
      return(NULL)
   })
    if (length(url) == 0) {
      p_id <- -file_counter
      platform <- "sona"
    } else {
      p_id <- url$participant
      if (is.null(url$platform)) {
        platform <- "sona"
      } else {
        platform <- url$platform
      }
    }
    data <- data %>%
      clean_names() %>%
      mutate(
        p_id = p_id,
        platform = platform,
        condition = unique(condition)[2],
        ended_on = as.factor(ended_on)
      )
    
    columns <- c(
      "sender", "timestamp", "p_id",
      "condition",  "block_no", "practice", "trial_no",
      "img_left", "img_right", "winner", "loser",
      "duration", "response", "ended_on", "platform"
    )
    if (
      any( !(columns %in% colnames(data)) )
    ) { 
      print(paste("Insufficient progress. Skipping", file_path))
      return(NULL)
    }
    data_length <- data %>%
      filter(sender == "trial", tolower(practice) == "false") %>%
      nrow()
    if (data_length == 0) { 
      print(paste("No trial data. Skipping", file_path))
      return(NULL)
    }

    data <- data %>%
      select(all_of(columns)) %>%
      mutate(
        timestamp = ymd_hms(timestamp, tz = "UTC"),
        tstamp_duration = as.numeric(
          difftime(timestamp, lag(timestamp), units = "auto")),
        ended_on = as.character(ended_on)
      ) %>%
      filter(
        sender == "trial",
        tolower(practice) == "false",
      ) %>%
      mutate(
        response = ifelse(ended_on == 'timeout' & !is.na(winner) & !is.na(loser),
                          ifelse(winner == img_right, 1, 
                                 ifelse(winner == img_left, 0, NA)),
                          response),
        ended_on = ifelse(ended_on == 'timeout' & !is.na(winner) & !is.na(loser),
                          'response', ended_on)
      ) %>%
      mutate(
        response = as.numeric(response),
        duration = round(as.numeric(duration)),
        tstamp_duration = round(tstamp_duration * 1000),
        trial_no = as.numeric(trial_no),
        block_no = as.numeric(block_no),
        p_id     = as.character(p_id),
        winner   = str_remove(winner, ".JPG"),
        loser    = str_remove(loser,  ".JPG"),
        img_left = str_remove(img_left, ".JPG"),
        img_right= str_remove(img_right,".JPG")
      ) %>%
      distinct(block_no, trial_no, .keep_all = TRUE)
    return(data)
  }
}

{
  if (read_data) {
    files <- list.files(here("pwc", "data", "raw"), pattern = ".*elanoma.*.csv$", full.names = TRUE)
    file_counter  <- 0
    data <- data.frame()
    unique_ids <- c(-999)
    for (file_path in files) {
      processed <- process_file(file_path, file_counter)
      if (is.null(processed)) {
        next
      }
      id <- unique(processed$p_id)
      if (id %in% unique_ids) {
        processed$p_id <- paste0(processed$p_id, "_", file_counter)
      }
      unique_ids <- c(unique_ids, id)
      file_counter <- file_counter + 1
      data <- bind_rows(data, processed)
    }
  }
}
  

if (save_raw) {
  write.csv(
    data,
    here("pwc", "data", "raw", "00_data_raw.csv"),
    row.names = FALSE
  )
}

######## Processing ----
# you have duplicate trials where two responses were made. I thought
# I had programmed the experiment response functions to allow only one. Oh well.
# You can elect to keep the first or remove them both. I have kept the first response.

# keep first
# data <- data %>% distinct(p_id, block_no, trial_no, .keep_all = TRUE)
# there's an odd trial with a negative duration. It's removed by 'remove all'
# data <- data %>%
#   filter(duration > 0)

# remove all, reverse score, remove stimulus load time from duration, generate a more reliable duration from screen timestamps.
feature_data <- data %>%
  group_by(p_id, block_no, trial_no) %>%
  # filter(n() == 1) %>%
  # ungroup() %>%
  mutate(
    duration = duration - 1500,
    tstamp_duration = tstamp_duration - 1500,
    response = if_else(
      condition %in% c("symmetry", "regular", "uniform"), 1 - response, response
    ),
    feature = case_when(
      condition %in% c("symmetry", "asymmetry") ~ "asymmetry",
      condition %in% c("regular", "irregular") ~ "border",
      condition %in% c("uniform", "colourful") ~ "colour"
    )
  ) %>%
  select(
    platform, p_id, feature, condition, block_no, trial_no,
    img_left, img_right, winner, loser, duration, response, ended_on,
    tstamp_duration
  ) %>%
  filter(tstamp_duration > 0) %>%
  group_by(feature) %>%
  do({
    if (save_processed) {
      write.csv(.,
        here("pwc", "data", "cleaned",
          paste0("btl_", unique(.$feature), ".csv")
        ),
        row.names = FALSE
      )
    }
    .
  }) %>%
  ungroup() %>%
  do({
    if (save_processed) {
      write.csv(.,
        here("pwc", "data", "cleaned", "btl_processed.csv"),
        row.names = FALSE
      )
    }
    .
  })
