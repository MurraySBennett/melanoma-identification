{
  rm(list=ls())
  library(BradleyTerry2)
  library(tidyverse)
  library(dplyr)
  library(qvalc)
  library(BradleyTerryScalable)
}

home = file.path(getwd(), "melanoma-identification", "feature-rating")
data_path = file.path(home, "btl-feature-data")


get_factors <- function(data){
  data$subject = factor(data$subject)
  combined_levels <- union(levels(as.factor(data$img_left)), levels(as.factor(data$img_right)))
  data$img_left <- factor(data$img_left, levels=combined_levels)
  data$img_right <- factor(data$img_right, levels=combined_levels)
  return(data)
}

set_unique_subject <- function(data){
  unique_s <- data %>%
    distinct(subject)
  unique_s$sequential <- seq_len(nrow(unique_s))
  data <- data %>%
    left_join(unique_s, by="subject")
  return(data)
}

get_data <- function(file_name, n_trials){
  file_name = file.path(data_path, file_name)
  data = read.csv(file_name)
  data = data[, c("img_left", "img_right", "response", "subject")]
  data = set_unique_subject(data)
  print(nrow(data))
  print(head(data))
  
  # 1 = left position, 0 = right position
  if (n_trials != FALSE) {
    data = head(data, n_trials)
  }
  data = get_factors(data)
  return(data)
}

group_wins <- function(data){
  df <- data %>%
    group_by(img_left, img_right) %>%
    summarise(left_wins=sum(!response), right_wins=sum(response))
  return(df)
}

position_model <-function(data){
  # random effect of subject and fixed effect of position (img_left)?
  model <- BTm(outcome=cbind(left_wins, right_wins), img_left, img_right, formula = ~image + (1|subject), data=data)
  return(model)
}

# from the CEMs data, pg10/11 of BradleyTerry2 Documentation
apply_model <- function(data){
  # .. represents the images and subject is a faactor specifying the student that made the comparison.
  # model<-BTm(outcome=cbind(left_wins, right_wins), player1 = img_left, player2 = img_right, formula = ~ .. + (1|subject), data=data)
  model <- BTm(outcome=1, player1=img_right, player2=img_left, forumla= response~img_left + img_right + (1|subject), data=data)
  return(model)
}

get_abilities <-function(data, label){
  model <- apply_model(data)
  print(summary(model))
  ability <- BTabilities(model)
  write.csv(ability, paste0("btl2-scores-",label,".csv"),row.names = FALSE)
}


scalable_model <- function(data){
  data <- data[c("img_left", "img_right", "response")]
  # data$item1wins <- abs(data$response - 1)
  # data$item2_wins <- data$response
  data$response <- ifelse(data$response == 0, "W1", "W2")
  coded <- codes_to_counts(data, c("W1", "W2"))
  bt_data <- btdata(coded)
  summary(bt_data)
  model <- btfit(bt_data, 1)
  abilities <- log(model$pi[1][1]) # you're trying to access the full list of abilities -- not sure if [1] will always work
  
}
n_trials = FALSE #5000
sym = get_data("btl-asymmetry.csv", n_trials)
bor = get_data("btl-border.csv", n_trials)
col = get_data("btl-colour.csv", n_trials)
# data <- data.frame(
#   winner= c("image1", "image2", "image3", "image4", "image5", "image8"),
#   loser = c("image2", "image3", "image5", "image6", "image7", "image3"),
#   subject = c(1, 1, 2, 2, 3, 3)
# )
# data<-get_factors(data)
# data <- group_wins(data)

# see page 12 of documentation
# feature.qv <- qvalc(BTabilities(model))
# plot(feature.qv)
# model <- position_model(data)
# print(summary(model))



# get_abilities(sym, "symmetry")

