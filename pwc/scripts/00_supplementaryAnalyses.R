{
  rm(list=ls())
  library(here)
  library(tidyverse)
  data <- read.csv(here("pwc", "data", "cleaned", "btl_processed_revised.csv"))
}

{
  too_fast <- 300
  data$response[
    data$ended_on == "timeout" |
      data$tstamp_duration < too_fast
  ] <- -1

  ##### participant x condition data ----
  condition_counts <- data %>%
    group_by(platform, feature, condition) %>%
    summarise(
      n.platform_cond = length(unique(p_id))
    ) %>%
    group_by(platform, feature) %>%
    mutate(n.platform_feature = sum(n.platform_cond)) %>%
    ungroup() %>%
    group_by(platform) %>%
    mutate(n.platform = sum(n.platform_feature / 2)) %>%
    ungroup() %>%
    group_by(feature) %>%
    mutate(n.feature = sum(n.platform_cond)) %>%
    ungroup() %>%
    mutate(n.total = sum(n.feature) / 4)

  ##### condition trial counts ----
  n_trials <- data %>%
    group_by(feature, condition) %>%
    summarise(cond.trial_count = n()) %>%
    group_by(feature) %>%
    mutate(feature.trial_count = sum(cond.trial_count))
}


{
  desc <- data %>%
    group_by(platform, p_id) %>%
    summarise(
      rt.min  = round(min(tstamp_duration), 0),
      rt.max  = round(max(tstamp_duration), 0),
      rt.over8 = sum(tstamp_duration > 8000),
      rt.mean = round(mean(tstamp_duration), 0),
      rt.sd   = round(sd(tstamp_duration), 0),
      tooSlow = sum(ended_on == "timeout"),
      tooFast = sum(tstamp_duration < too_fast),
      resp.r  = sum(response == 1) / n(),
      resp.l  = sum(response == 0) / n(),
      bias    = round(resp.r - 0.5, 2),
      badBias = ifelse(resp.r > 0.6 | resp.l > 0.6, 1, 0),
      n = n()
    )

  desc$platform <- as.factor(desc$platform)
  
  #### check different RT cut-offs ----
  for (i in seq(200, 1000, 100)) { 
    n_removed <- sum(data$duration < i)
    print(paste(i, n_removed))
  }
  print(1 - sum(desc$rt.over8) / sum(desc$n))
}

{
  # end of block warning if n(rt) > 10 or bias > 0.75 (here, > abs(0.25))
  resp_warnings <- data %>%
    group_by(platform, p_id, block_no) %>%
    summarise(
      rt    = sum(tstamp_duration < 300),
      bias  = round(abs((sum(response == 1) / n()) - 0.5), 2)
    )

  block_warnings <- respWarnings %>%
    group_by(p_id) %>%
    summarise(
      n = sum(bias > 0.25)
    )
  print(sum(blockWarnings$n > 3) / length(blockWarnings))
}

{
  #### Platform comparisons ----
  anova.rt    <- aov(rt.mean ~ platform, data = desc)
  anova.bias  <- aov(bias ~ platform, data = desc)

  summary(anova.rt)
  summary(anova.bias)
}


{
  #### describe estimates ----
  est <- read.csv(here("pwc", "data", "estimates", "btl_cv_data_revised.csv"))

  desc_est <- est %>%
    select(pi_sym, pi_bor, pi_col) %>%
    summarise(
      a.min = round(min(pi_sym, na.rm = TRUE), 3),
      b.min = round(min(pi_bor, na.rm = TRUE), 3),
      c.min = round(min(pi_col, na.rm = TRUE), 3),
      a.max = round(max(pi_sym, na.rm = TRUE), 3),
      b.max = round(max(pi_bor, na.rm = TRUE), 3),
      c.max = round(max(pi_col, na.rm = TRUE), 3),
      a.med = round(median(pi_sym, na.rm = TRUE), 3),
      b.med = round(median(pi_bor, na.rm = TRUE), 3),
      c.med = round(median(pi_col, na.rm = TRUE), 3),
    ) %>%
    print(.)
  summary(est[c("pi_sym", "pi_bor", "pi_col")])
}
