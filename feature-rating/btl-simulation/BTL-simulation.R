library(tictoc)
library(rstan)
library(igraph)
library(furrr)

rstan_options(auto_write=TRUE)
no_cores <- availableCores() - 1
options(mc.cores = parallel::detectCores(logical = FALSE))

setwd("C:/Users/qlm573/melanoma-identification/feature-rating/btl-simulation")
save_data = TRUE

##########
# You will want to toggle:
# n_simulations: iterations per level
# min_trials and n_participants: you have a constant trial count of 450 per participant. Therefore, the total number of trials is relative to this hyperparameter
# min_ and max_ players: the number of images to be included. A maximum value of 71ish thousand.

##########
# Returns:
# correlation between predicted and actual alpha
# Chi^2 test results between expected and observed alpha
# ranking error estimates
# rmse
# graph connectivity
# proportion of n_players accessed
# strength of connectivity ?


run_bay = FALSE
#per iteration:
#   both: ~13s
#   MLE: ~0.12s
#   Bay: ~13s

n_simulations = 1000

min_participants <- 40
max_participants <- 80 # you want to overestimate this, to see how far back you need to go.
participant_step <- 10

trials_per_participant <- 450
min_trials <- trials_per_particiapant * min_participants # 450 trials over ~25mins - this is a kind of proxy for n_participants
max_trials <- trials_per_particiapant * max_participants
n_trials <- seq(min_trials, max_trials, trials_per_particiapant * participant_step)#participant_step * trials_per_particiapant)



min_players <- 5000 # players === images
max_players <- 70000# 40000
n_players <- seq(min_players, max_players, min_players) # you might want to make a specific step variable


total_sims <- length(n_trials) * length(n_players) * n_simulations
# sim_list <- list(rep(1:n_simulations, length(n_players) * length(n_trials)))
# trial_list<-list(rep(n_trials, length(n_players), each=n_simulations))
# player_list<-list(rep(n_players, 1, each=n_simulations*length(n_trials)))

sim_col_names <- c("n_trials", "n_players", "sim_no", "sp_rho", "rmse", "connected", "n_components", "mu_dist", "max_dist", "e_density", "diam", "med_deg", "min_deg", "max_deg")
sim_data <- data.frame(matrix(ncol=length(sim_col_names), nrow=total_sims, dimnames = list(NULL, sim_col_names)))



get_contests <- function(p_list, trials){
  players <- sample(p_list, trials * 2, replace = T)
  contests <- matrix(players, nrow = trials, byrow = T)
  
  no_mates <- contests[, 1] == contests[, 2]
  while (sum(no_mates > 0)){
    contests[no_mates, 1] <- sample(p_list, sum(no_mates), replace=T)
    no_mates <- contests[, 1] == contests[, 2]
  }
  return( contests )
}

inv_logit <- function(u){
  inv_log <- 1 / (1 + exp(-u))
  return ( inv_log )
}

center <- function(u){
  c <- u - sum(u) / length(u)
  return ( c )
}

simulate_data <- function(n_players, n_trials, alpha){
  sim_col_names <- c("player0", "player1", "winner")
  players <- seq(1, n_players)
  df <- data.frame(matrix(ncol=3, nrow=n_trials, dimnames=list(NULL, sim_col_names)))
  
  df[, 1:2] <- get_contests(players, n_trials)
  
  log_odds_p1 <- alpha[df$player1] - alpha[df$player0]
  win_prob_p1 <- inv_logit(log_odds_p1)
  
  y <- rbinom(n_trials, 1, win_prob_p1)
  df$winner <- y
  
  return ( df )
}

get_edges <- function(df){
  # remove winner column, convert to array of vertex pairs (edges)
  # undirected graph - you just want to see if players were in contests. The results would make it a directed graph
  
  edges <- as.vector(t(as.matrix(df[-3])))
}

get_graph_desc <- function(graph){
  fully_connected <- is_connected(graph)
  n_components <- components(graph)$no
  mu_dist <- mean_distance(graph, directed=FALSE)
  max_dist <- max(distances(graph))
  e_density <- edge_density(graph, loops=FALSE)
  # greatest distance between a pair of items -- the longest shortest distance between two items.
  diam <- diameter(graph)
  # connections between items
  deg <- degree(graph, mode="all")
  med_deg <- median(deg)
  min_deg <- min(deg)
  max_deg <- max(deg)
  
  return ( c(fully_connected, n_components, mu_dist, max_dist, e_density, diam, med_deg, min_deg, max_deg) )
}

nDCG <- function(relevance){
  # Normalized Discounted Cumulative Gain
  # I think relevance would be best used here as the difference 
  # between predicted and actual rank (+1). Must be +1 because
  # rel = c(0,0,0) and c(1,1,1) both output 1, which is incorrect.
  # Perfect prediction = 1
  p <- length(relevance)
  dcg <- sum(relevance[seq(p)] / log(seq(p) + 1, 2))
  i_dcg <- sum((2^(relevance[seq(p)]) - 1) / log(seq(p) + 1, 2))
  n_dcg <- dcg / i_dcg

  return ( n_dcg )
}

get_desc <- function(actual, predicted) {
  sp_rho <- NA
  tryCatch({
    sp_rho <- cor(actual, predicted, method = "spearman", use="complete.obs")
  }, error=function(e){}
  )
  # rank error doesn't represent the error of individual ability predictions. 
  # relevance <- sort(abs(predicted - actual)) + 1 / length(predicted)
  # ndcg <- nDCG(relevance)
  rmse <- sqrt(mean((actual-predicted)^2))
  
  return ( c(sp_rho, rmse) )
}


if (run_bay){
  bayes_model <- stan_model("bayesian-ind.stan")
}
mle_model   <- stan_model("individual-uniform.stan")

tic()


sim_counter <- 1
# Run simulations
for (t in seq_along(n_trials)){
  for (p in seq_along(n_players)){
    for (s in 1:n_simulations){
      alpha <- center(rnorm(n_players[p]))
      df <- simulate_data(n_players[p], n_trials[t], alpha)

      model_data <-
        list(K = n_players[p],
             N = n_trials[t],
             player0 = df$player0,
             player1 = df$player1,
             y = df$winner
             )

      mle_est <- optimizing(mle_model, data=model_data)
      if (run_bay) {
        ind_post<- sampling(bayes_model, data=model_data)
      }

      alpha_star  <- mle_est$par[paste("alpha[", 1:n_players[p], "]", sep="")]
      if (run_bay) {
        alpha_hat   <- rep(NA, n_players[p])
        for (player in 1:n_players[p]){
          alpha_hat[player] <- mean(extract(ind_post)$alpha[ , player])
        }
        ranked_bay <- extract(ind_post)$ranking
      }

      ranked_mle <- mle_est$par[paste("ranked[", 1:n_players, "]",sep="")]
      rank_actual <- abs(rank(alpha) - n_players[p]) + 1 # highest alpha to lowest
      rank_pred <- abs(rank(mle_est$par[paste("alpha[", 1:n_players, "]",sep="")]) - n_players[p])

      gr <- graph(edges=get_edges(df), n=n_players[p])

      sim_data[sim_counter, ] <- c(n_trials[t], n_players[p], s, get_desc(rank_actual, rank_pred), get_graph_desc(gr) )

      sim_counter = sim_counter + 1

    }
  }
}
toc()

if (save_data){
  write.csv(sim_data, paste('.\\simulation_p', min_participants, '-', max_participants, '_images', min_players, '-', max_players, '.csv',sep=""), row.names=FALSE)
}















