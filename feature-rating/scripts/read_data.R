## packages
{
  library(rstan)
  library(tidyverse)
}

## functions
process_data <- function(f){
  data_columns = c(
    'sender', 'timestamp', 'pID', 
    'condition',  'blockNo', 'practice', 'trialNo', 
    'img_left', 'img_right', 'winner', 'loser', 
    'duration', 'response', 'ended_on')
  
  df = read.csv(f)
  condition = unique(df$condition)[2]
  
  # this line removes any extraneous text that may exist (definitely for the first 5 participants)
  pID = na.omit(as.numeric(gsub("[^0-9.]+", "", unique(df$url))))
  
  if (file.info(f)$size < 1000){
    print(f)
    return(NULL)
  }
  # filter practice trials and transition screens, retaining only trial data
  df = df[df$sender=="trial" & df$practice=="false", ]
  
  if (nrow(df) == 0){
    return(NULL)
  } else {
    
    # identify experiment information
    df$condition = condition
    df$pID = pID
    df$response=as.numeric(df$response)
    # check experiment duration
    # exp_duration = sum(df$duration)/1000/60
    # print(exp_duration)
    # print(mean(df$duration) / 1000 * 400 / 60)
    
    # remove duplicate trials
    df = df[!(duplicated(df[,c('blockNo','trialNo')]) | duplicated(df[,c("blockNo","trialNo")],fromLast=TRUE)),]
    # filter columns
    df = df[, data_columns]

    # add total trial count if desired (1-300, rather than 1-50 repeated)
    # df = df %>%
    #   group_by(blockNo) %>%
    #   mutate(total_trial = row_number()) %>%
    #   ungroup()
    
    df$total_trial = seq(1,nrow(df),1)
    return(df)
  }
}

## project management
{
  path <- list(home='C:/Users/qlm573/melanoma-identification/feature-rating/')
  path$data = paste0(path$home, 'experiment/melanoma-2afc/data/')
  path$figures = paste0(path$home, 'figures')
  path$model = paste0(path$home, 'btl-model')
}


## read data
{
  setwd(path$data)
  files = list.files()
  data = bind_rows(lapply(files, process_data))

  data <- data %>%
    group_by(condition, pID) %>%
    mutate(pnum = cur_group_id())
  setwd(path$home)
}

# for (i in unique(data$pID)){
#   print(i)
# }
# print(length(unique(data$pID)))

## summary data
{
  summary = data %>% 
    group_by(condition, pnum) %>%
    summarise(
      pos_bias=mean(response, na.rm=TRUE),
      count_left=sum(response==0, na.rm=TRUE),
      count_right=sum(response==1, na.rm=TRUE),
      count_timeouts=sum(is.na(response)),
      meanRT=(mean(duration)-1500)/1000, # seconds
      semRT=sd(duration) / sqrt(length(duration)),
      totalT=(sum(duration)/1000/60), # minutes
      ntrials=(length(duration))
    )
}


## model data
{
  n_images = length(unique(c(data$img_left, data$img_right)))
  setwd(path$model)
  mle_model <- stan_model("individual-uniform.stan")
  model_data <-
    list(K = n_images,
         N = length(data$img_left), # number of trials
         player0 = data$img_left,
         player1 = data$img_right,
         y = data$winner
    )
  mle_est <- optimizing(mle_model, data=model_data)
  setwd(path$home) 
}
## plotting
# plt_meanRT = ggplot(df_desc, aes(pnum, meanRT)) + 
#   geom_point()
# plt_expdur = ggplot(df_desc, aes(pnum, totalT)) + 
#   geom_point()
# plt_meanRT
# plt_expdur
