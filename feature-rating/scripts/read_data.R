## packages
{
  library(tidyverse)
}

## functions
process_data <- function(file){
  data_columns = c(
    'sender', 'timestamp', 'pID', 
    'condition',  'blockNo', 'practice', 'trialNo', 
    'img_left', 'img_right', 'winner', 'loser', 
    'duration', 'response', 'ended_on')
  
  df = read.csv(file)
  condition = unique(df$condition)[2]
  
  # this line removes any extraneous text that may exist (definitely for the first 5 participants)
  pID = na.omit(as.numeric(gsub("[^0-9.]+", "", unique(df$url))))
  
  # filter practice trials and transition screens, retaining only trial data
  df = df[df$sender=="trial" & df$practice=="false", ]
    
  if (nrow(df) == 0){
    return(NULL)
  } else {
    
    # identify experiment information
    df$condition = condition
    df$pID = pID
    
    # check experiment duration
    # exp_duration = sum(df$duration)/1000/60
    # print(exp_duration)
    # print(mean(df$duration) / 1000 * 400 / 60)
    
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
  path$figures = paste0(path$home, '/figures')
}


## read data
{
  setwd(path$data)
  files = list.files()
  data = bind_rows(lapply(files, process_data))

  data <- data %>%
    group_by(pID) %>%
    mutate(pnum = cur_group_id())
}

# for (i in unique(data$pID)){
#   print(i)
# }
# print(length(unique(data$pID)))

## summary data
{
  df_desc = data %>% 
    group_by(pnum) %>%
    summarise(
      meanRT=(mean(duration)-1500)/1000, # seconds
      totalT=(sum(duration)/1000/60)) # minutes
}


## plotting
plt_meanRT = ggplot(df_desc, aes(pnum, meanRT)) + 
  geom_point()
plt_expdur = ggplot(df_desc, aes(pnum, totalT)) + 
  geom_point()
plt_meanRT
plt_expdur
