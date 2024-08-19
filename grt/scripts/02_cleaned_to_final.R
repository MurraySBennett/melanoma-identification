## load packages ----
rm(list = ls())
library(tidyverse)
library(here)

## read data ----
files <- list.files(
  path = here("grt", "data", "cleaned"),
  pattern = "*.csv"
)

# data checks ---------
# note that the processing of fast RTs is done in 01_raw_to_cleaned

data <- bind_rows(
  lapply(
    here("grt", "data", "cleaned", files), read.csv
  )
)
# possible values: 0.375 # (0.5+0.25+0.25+0.5) / 4 = 0.375
min_acc <- 0.3
min_rt  <- 1000

data %>%
  group_by(condition, pID) %>%
  summarise(
    acc = mean(correct)
  ) %>%
  filter(acc < min_acc) %>%
  ggplot(aes(x = pID, y = acc)) +
  geom_point() +
  theme_classic()

data <- data %>%
  group_by(condition, pID) %>%
  filter(
    !mean(correct, na.rm = TRUE) <= min_acc,
    !mean(rt, na.rm = TRUE) <= min_rt
  )

condition_counts <- data %>%
  group_by(condition, pID) %>%
  summarise(n = n()) %>%
  group_by(condition) %>%
  summarise(n = n())

print(condition_counts)

write.csv(data, here("grt", "data", "final", "mel_grt_data.csv"))
