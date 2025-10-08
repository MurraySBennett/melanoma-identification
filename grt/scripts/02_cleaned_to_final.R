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
min_acc <- 0.5 #0.3
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

write.csv(data, here("grt", "data", "final", paste0("mel_grt_data_min_acc_", min_acc, ".csv")))


cms <- list(ab=get_cm(data), ac=get_cm(data), bc=get_cm(data))

{
  summary_data <- data %>%
    filter(response != "") %>%
    group_by(pID, condition, stimulus) %>%
    summarise(
      accuracy_per_participant = mean(correct),
      .groups = 'drop'
    ) %>%
    group_by(condition, stimulus) %>%
    summarise(
      mean_accuracy = mean(accuracy_per_participant),
      se_accuracy = sd(accuracy_per_participant) / sqrt(n()),
      .groups = 'drop'
    ) %>%
    mutate(stimulus = factor(stimulus, levels = c("ll", "hl", "lh", "hh")))
  
  global_summary_data <- data %>%
    filter(response != "") %>%
    group_by(pID, condition) %>%
    summarise(
      accuracy_per_participant_overall = mean(correct),
      .groups = 'drop'
    ) %>%
    group_by(condition) %>%
    summarise(
      global_mean = mean(accuracy_per_participant_overall),
      global_se = sd(accuracy_per_participant_overall) / sqrt(n()),
      .groups = 'drop'
    )
  
  ggplot(summary_data, aes(x = stimulus, y = mean_accuracy, fill = stimulus)) +
    geom_errorbar(aes(ymin = mean_accuracy - se_accuracy, ymax = mean_accuracy + se_accuracy),
                  width = 0.2, position = position_dodge(width = 0.9), color = "darkgray", linewidth = 0.8) +
    geom_point(stat = "identity", position = position_dodge(width = 0.9),
               size = 3, shape = 21, fill = "skyblue", color = "darkgray", stroke = 0.1) +
    
    geom_errorbar(data = global_summary_data,
                  aes(x = 2.5, ymin = global_mean - global_se, ymax = global_mean + global_se),
                  inherit.aes = FALSE,
                  width = 0.3, color = "darkgray", linewidth = 0.8) +
    geom_point(data = global_summary_data,
               aes(x = 2.5, y = global_mean),
               inherit.aes = FALSE,
               size = 3, shape = 23, fill = "salmon", color = "darkgray", stroke = 0.1) +
    facet_wrap(~ condition, scales = "fixed", ncol = 3, labeller = as_labeller(toupper)) +
    labs(
      x = "Stimulus level",
      y = "Mean Accuracy"
    ) +
    scale_y_continuous(limits = c(0, 1), breaks = c(0, 0.25, 0.5, 0.75, 1)) +
    scale_x_discrete(labels = toupper) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
      axis.title.x = element_text(size = 12),
      axis.title.y = element_text(size = 12),
      axis.text.x = element_text(hjust = 1, size = 10),
      axis.text.y = element_text(size = 10),
      strip.text = element_text(size = 12, face = "bold"),
      
      legend.position = "none",
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      axis.line = element_line(colour = "black"),
      aspect.ratio = 1
    )
  
  ggsave(here("grt", "figures", paste0("stimulus_acc_", min_acc, ".png")), width = 10, height = 4)
  
}

{
  
  response_frequency_data <- data %>%
    filter(response != "") %>% 
    group_by(pID, condition) %>% 
    mutate(total_trials_per_pid_condition = n()) %>%
    ungroup() %>% 
    group_by(pID, condition, response) %>% 
    summarise(
      count_response = n(), 
      total_trials = first(total_trials_per_pid_condition), 
      .groups = 'drop'
    ) %>%
    mutate(
      frequency = count_response / total_trials 
    )
  
  bias_summary_data_simple <- response_frequency_data %>%
    group_by(condition, response) %>%
    summarise(
      mean_frequency = mean(frequency),
      se_frequency = sd(frequency) / sqrt(n()),
      .groups = 'drop'
    ) %>%
    mutate(response = factor(response, levels = c("ll", "hl", "lh", "hh")))
  
  response_freq_plot <- ggplot(bias_summary_data_simple, aes(x = response, y = mean_frequency, fill = response)) +
    geom_point(stat = "identity", position = position_dodge(width = 0.9),
               size = 3, shape = 21, fill = "green", color = "black", stroke = 0.5) + 
    geom_errorbar(aes(ymin = mean_frequency - se_frequency, ymax = mean_frequency + se_frequency),
                  width = 0.2, position = position_dodge(width = 0.9), color = "black", linewidth = 0.8) +
    
    facet_wrap(~ condition, scales = "fixed", ncol = 3, labeller = as_labeller(toupper)) +
    labs(
      title = "Overall Response Frequencies by Condition", 
      x = "Response Type", 
      y = "Mean Response Frequency"
    ) +
    scale_y_continuous(limits = c(0, 1), breaks = c(0, 0.25, 0.5, 0.75, 1)) +
    scale_x_discrete(labels = toupper) + 
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
      axis.title.x = element_text(size = 12),
      axis.title.y = element_text(size = 12),
      axis.text.x = element_text(angle = 45, hjust = 1, size = 10), 
      axis.text.y = element_text(size = 10),
      strip.text = element_text(size = 12, face = "bold"),
      
      legend.position = "none", 
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      axis.line = element_line(colour = "black"),
      aspect.ratio = 1
    )
  
  print(response_freq_plot)
  ggsave(here("grt", "figures", paste0("response_frequencies_overall_", min_acc, ".png")), plot = response_freq_plot, width = 10, height = 4) 
}
