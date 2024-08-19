## load packages ----
rm(list = ls())
library(tidyverse)
library(here)
library(skimr)
library(janitor)
## read data ----

files <- list.files(
  path = here("grt", "data", "raw"),
  pattern = "*.csv"
)
participant_counter <- 1

get_condition <- function(x) {
  chars <- sort(unique(x))[2]
  if (grepl("s", chars) && grepl("b", chars)) {
    condition <- "ab"
  } else if (grepl("s", chars) && grepl("c", chars)) {
    condition <- "ac"
  } else if (grepl("b", chars) && grepl("c", chars)) {
    condition <- "bc"
  }
  return(condition)
}



for (f in files) {
  data <- read.csv(here("grt", "experiment", "data", "raw", f))
  data <- data %>%
    clean_names() %>%
    rename(rt = duration, stimulus = correct_response) %>%
    mutate(
      pID       = participant_counter,
      condition = get_condition(condition),
      age       = sort(unique(age)),
      gender    = sort(unique(gender))[2],
      correct   = tolower(correct) == "true",
    ) %>%
    filter(
      sender == "trial",
      tolower(practice) == "false",
      tolower(staircase) == "false"
    ) %>%
    select(
      pID, condition, block_no, trial_no, #practice, staircase,
      stimulus, response, correct, rt, age, gender
    )
  # the images are revealed after 1000ms to allow them time to load.
  # I account for this here to only provide response times from when
  # the participant can see the images.
  data$rt <- round(data$rt, 3) - 1000
  data$rt[data$rt <= 200] <- NA
  data$correct[data$rt <= 200] <- FALSE
  file_id <- paste0(
    "participant_", participant_counter, "_", data$condition[1], ".csv"
  )
  write_csv(data, here("grt", "data", "cleaned", file_id))
  participant_counter <- participant_counter + 1
}
