## packages
{
  rm(list=ls())
  library(BradleyTerry2)
  # library(BradleyTerryScalable)
  library(reshape2)
  library(tidyverse)
  library(igraph)

  library(rstan)
  rstan_options(auto_write=TRUE) # stops rstan recompiling unchanged stan programs
  # no_cores <- availableCores() - 1
  # options(mc.cores = parallel::detectCores())
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

    df = na.omit(df)
    # add total trial count if desired (1-300, rather than 1-50 repeated)
    # df = df %>%
    #   group_by(blockNo) %>%
    #   mutate(total_trial = row_number()) %>%
    #   ungroup()
    
    df$total_trial = seq(1,nrow(df),1)
    return(df)
  }
}

to_cmat <- function(df){
  img_left = df$img_left
  img_right = df$img_right
  response = df$response
  
  # Get unique image IDs and order them
  img_ids <- sort(unique(c(img_left, img_right)))
  
  # Create empty matrix with ordered row and column names
  img_matrix <- matrix(0, nrow=length(img_ids), ncol=length(img_ids), dimnames=list(img_ids, img_ids))
  
  # Fill in matrix with win counts
  for (i in seq_along(response)) {
    if (response[i] == 1) {
      img_matrix[img_right[i], img_left[i]] <- img_matrix[img_right[i], img_left[i]] + 1
    } else if (response[i] == 0) {
      img_matrix[img_left[i], img_right[i]] <- img_matrix[img_left[i], img_right[i]] + 1
    }
  }
  return(img_matrix)
}

to_long <- function(cmat){
  df <- melt(cmat)
  colnames(df) <- c("img1", "img2", "wins")
  df <- as.data.frame(df)
  
  # create new data frame
  df2 <- data.frame(img1 = character(),
                    img2 = character(),
                    win1 = integer(),
                    win2 = integer(),
                    stringsAsFactors = FALSE)
  
  # loop through combinations of images
  for(i in 1:nrow(df)) {
    for(j in 1:nrow(df)) {
      # skip if comparing same img
      if(df[i, "img1"] == df[j, "img2"]) {
        next
      }
      # compare img1 to img2
      if(df[i, "img1"] == df[j, "img1"]) {
        new_row <- data.frame(journal1 = df[i, "img2"],
                              journal2 = df[j, "img2"],
                              win1 = df[i, "wins"],
                              win2 = df[j, "wins"])
        df2 <- rbind(df2, new_row)
      }
    }
  }
  
  # reset row names
  row.names(df2) <- NULL
  return(df2)
}

id_to_int <- function(img_left, img_right){
  # Extract the numeric part of the IDs
  id_left <- as.integer(gsub("[^0-9]", "", img_left))
  id_right <- as.integer(gsub("[^0-9]", "", img_right))
  
  # Return the integer columns as a new data frame
  int_data <- data.frame(left_id = id_left, right_id = id_right)
  
  return(int_data)
}

set_uniqueID <- function(img_left, img_right) {
  # Combine the two columns and extract the numeric part of the IDs
  all_ids <- gsub("[^0-9]", "", c(img_left, img_right))
  
  # Map each numeric ID to a unique integer value
  id_map <- setNames(seq_along(unique(all_ids)), unique(all_ids))
  
  # Convert the IDs to integer values using the lookup table
  img_left_int <- id_map[gsub("[^0-9]", "", img_left)]
  img_right_int <- id_map[gsub("[^0-9]", "", img_right)]
  
  # Combine the integer IDs and corresponding string IDs into a data frame
  id_data <- data.frame(img_left_id = img_left_int, img_left = img_left, 
                        img_right_id = img_right_int, img_right = img_right)
  
  return(id_data)
}

get_combination_wins <- function(data) {
  # Create a new column indicating which image was selected
  data$selected_img <- ifelse(data$response == 1, data$img_right, data$img_left)
  
  # Group the data by each combination of images and count the number of wins
  wins_data <- data %>%
    group_by(img_left, img_right) %>%
    summarize(wins = sum(selected_img == img_right & response == 1) +
                sum(selected_img == img_left & response == 0))
  
  # Return the wins data
  return(wins_data)
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
  
  # int_cols <- id_to_int(data$img_left, data$img_right)
  # data <- cbind(data, int_cols)
  
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

## subset data 
{
  data_reg = data[data$condition=='regular', c("img_left", "img_right", "response")]
  # Convert string IDs to integer IDs
  reg_id <- set_uniqueID(data_reg$img_left, data_reg$img_right)
  # data_reg = data[data$condition=='regular', c("winner", "loser")]
  # Add integer IDs to original data frame
  data_reg$left_id <- reg_id$img_left_id
  data_reg$right_id <- reg_id$img_right_id
  
  
  # data_irr = data[data$condition=='irregular', ] 
  # irr_id <- set_uniqueID(data_irr$img_left, data_irr$img_right)
  # data_irr$left_id <- irr_id$img_left_id
  # data_irr$right_id <- irr_id$img_right_id
}


## stan model data
{
  # n_images = length(unique(c(data_reg$img_left, data_reg$img_right)))
  # n_trials = length(data_reg$img_left)
  # 
  # setwd(path$model)
  # mle_model <- stan_model("individual-uniform.stan")
  # model_data <-
  #   list(K = n_images,
  #        N = n_trials, # number of trials
  #        player0 = data_reg$left_id,
  #        player1 = data_reg$right_id,
  #        y = data_reg$response
  #   )
  # mle_est <- optimizing(mle_model, data=model_data)
  # alpha_star  <- mle_est$par[paste("alpha[", 1:n_images, "]", sep="")]
  # ranked_mle <- mle_est$par[paste("ranked[", 1:n_images, "]",sep="")]
  # rank_pred <- abs(rank(mle_est$par[paste("alpha[", 1:n_images, "]",sep="")]) - n_images)
  # 
  # setwd(path$home)
}


## bradley terry 2 model - journal example format
{
  # data_reg <- data
  # data_reg[data_reg$condition=='irregular', 'response'] = abs(data_reg[data_reg$condition=='irregular', 'response'] - 1)
  
  # img_matrix = to_cmat(data_reg)
  
  # data_long <- to_long(img_matrix)
  
  # bt2_model <- BTm(cbind(wins1, wins2), img1, img2, ~ img,
                     # id = "img", data = data_long)
   
}


## bradley terry 2 model - lizard example format
{
  # data_reg <- data[data$condition=='regular', ]
  # bt2_winner_loser <- data_reg[, c("winner", "loser")]
  # bt2_wl_modle <- BTm(1, winner, loser,
  #                 data = bt2_winner_loser)
}


## bradley terry scalable
{
  
  data_reg <- data
  data_reg[data_reg$condition=='irregular', 'response'] = abs(data_reg[data_reg$condition=='irregular', 'response'] - 1)
  # img_left = data_reg$img_left
  # img_right = data_reg$img_right
  # response = data_reg$response
  # 
  # # Get unique image IDs and order them
  # img_ids <- sort(unique(c(img_left, img_right)))
  # 
  # # Create empty matrix with ordered row and column names
  # img_matrix <- matrix(0, nrow=length(img_ids), ncol=length(img_ids), dimnames=list(img_ids, img_ids))
  # 
  # # Fill in matrix with win counts
  # for (i in seq_along(response)) {
  #   if (response[i] == 1) {
  #     img_matrix[img_right[i], img_left[i]] <- img_matrix[img_right[i], img_left[i]] + 1
  #   } else if (response[i] == 0) {
  #     img_matrix[img_left[i], img_right[i]] <- img_matrix[img_left[i], img_right[i]] + 1
  #   }
  # }
  img_matrix = to_cmat(data_reg)
  
  # check how many times i beat j via: img_matrix[i, j]
  bts_data <- btdata(img_matrix, return_graph = TRUE) 
  summary(bts_data)
  bts_fit = btfit(bts_data, a=1) # 1 if fully connect, > 1 if not fully connected which runs a bayesian method
  summary(bts_fit)
  # coef(bts_fit)
 }

## plotting
# plt_meanRT = ggplot(df_desc, aes(pnum, meanRT)) + 
#   geom_point()
# plt_expdur = ggplot(df_desc, aes(pnum, totalT)) + 
#   geom_point()
# plt_meanRT
# plt_expdur
