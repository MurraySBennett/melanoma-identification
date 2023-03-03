library(tictoc)
library(rstan)
rstan_options(auto_write=TRUE)
options(mc.cores = parallel::detectCores(logical = FALSE))

run_bay = FALSE
#per iteration:
#   both: ~13s
#   MLE: ~0.12s
#   Bay: ~13s

n_simulations = 1 # 1000?

n_participants <- 1 # you want to overestimate this, to see how far back you need to go.

min_trials <- 450 # 450 trials over ~25mins - this is a kind of proxy for n_participants
max_trials <- 450 #min_trials * n_participants
n_trials <- seq(min_trials, max_trials, min_trials)

min_players <- 100
max_players <- 100
n_players <- seq(min_players, max_players, min_players) # you might want to make a specific step variable


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

if (run_bay){
  bayes_model <- stan_model("bayesian-ind.stan")
}
mle_model   <- stan_model("individual-uniform.stan")

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
    }
  }
}























