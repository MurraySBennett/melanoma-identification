
## Libraries and setup ------
{
  rm(list = ls())
  library(tictoc)
  library(rstan)
  library(igraph)
  library(furrr)
  library(here)

  rstan_options(auto_write = TRUE)
  no_cores <- availableCores() - 1
  options(mc.cores = parallel::detectCores())

  save_data <- TRUE
}

##########
# You will want to toggle:
# n_sims: iterations per level
# min_trials and n_participants: you have a constant trial count of 450 per
# participant. Therefore, the total number of trials is relative to this
# hyperparameter min_ and max_ players: the number of images to be included.
# A maximum value of 71ish thousand.

##########
# Returns:
# correlation between predicted and actual alpha
# Chi^2 test results between expected and observed alpha
# ranking error estimates
# rmse
# graph connectivity
# proportion of n_players accessed
# strength of connectivity ?

## hyperparameters ------
{
  run_bay <- FALSE
  # per iteration:
  #   both: ~13s
  #   MLE: ~0.12s
  #   Bay: ~13s

  n_sims  <- 100
  scaling <- 100
  max_p   <- 250
  min_p   <- 10
  p_step  <- 10

  p_trials    <- 450 # proxy for n_participants
  min_trials  <- p_trials * min_p
  max_trials  <- p_trials * max_p
  n_trials    <- round(
    seq(min_trials, max_trials, p_trials * p_step) / scaling
  )

  max_players <- 75000
  min_players <- 5000 # players === images
  player_step <- 5000
  n_players <- round(
    seq(min_players, max_players, player_step) / scaling
  )

  total_sims <- length(n_trials) * length(n_players) * n_sims

  sim_col_names <- c(
    "n_participants", "n_trials", "n_players", "sim_no",
    "sp_rho", "sp_p", "rmse", "connected", "n_components",
    "mu_dist", "max_dist", "e_density", "diam",
    "med_deg", "min_deg", "max_deg"
  )


  sim_data <- data.frame(
    matrix(
      ncol = length(sim_col_names),
      nrow = total_sims,
      dimnames = list(NULL, sim_col_names)
    )
  )
}

## functions ------
{
  get_contests <- function(p_list, trials) {
    players   <- sample(p_list, trials * 2, replace = TRUE)
    contests  <- matrix(players, nrow = trials, byrow = TRUE)

    # check for those playing with themselves (i.e., no mates...)
    no_mates <- contests[, 1] == contests[, 2]
    while (sum(no_mates > 0)) {
      contests[no_mates, 1] <- sample(p_list, sum(no_mates), replace = TRUE)
      no_mates <- contests[, 1] == contests[, 2]
    }
    return(contests)
  }


  inv_logit <- function(u) {
    return(1 / (1 + exp(-u)))
  }


  center <- function(u) {
    return(u - sum(u) / length(u))
  }


  simulate_data <- function(n_players, n_trials, alpha) {
    sim_col_names <- c("player0", "player1", "winner")
    players       <- seq(1, n_players)
    df <- data.frame(
      matrix(
        ncol = 3,
        nrow = n_trials,
        dimnames = list(NULL, sim_col_names)
      )
    )

    df[, 1:2]   <- get_contests(players, n_trials)
    log_odds_p1 <- alpha[df$player1] - alpha[df$player0]
    win_prob_p1 <- inv_logit(log_odds_p1)
    df$winner   <- rbinom(n_trials, 1, win_prob_p1)
    return(df)
  }


  get_edges <- function(df) {
    edges <- as.vector(t(as.matrix(df[-3])))
    return(edges)
  }


  get_graph_desc <- function(graph) {
    fully_connected <- is_connected(graph)
    n_components    <- components(graph)$no
    mu_dist         <- mean_distance(graph, directed = FALSE)
    max_dist        <- max(distances(graph))
    e_density       <- edge_density(graph, loops = FALSE)
    diam            <- diameter(graph)
    deg             <- degree(graph, mode = "all")
    med_deg         <- median(deg)
    min_deg         <- min(deg)
    max_deg         <- max(deg)
    return(c(
      fully_connected, n_components,
      mu_dist, max_dist, e_density, diam,
      med_deg, min_deg, max_deg
    ))
  }


  nDCG <- function(relevance) {
    # Normalized Discounted Cumulative Gain
    # I think relevance would be best used here as the difference 
    # between predicted and actual rank (+1). Must be +1 because
    # rel = c(0,0,0) and c(1,1,1) both output 1, which is incorrect.
    # Perfect prediction = 1
    p     <- length(relevance)
    dcg   <- sum(relevance[seq(p)] / log(seq(p) + 1, 2))
    i_dcg <- sum((2^(relevance[seq(p)]) - 1) / log(seq(p) + 1, 2))
    n_dcg <- dcg / i_dcg
    return(n_dcg)
  }


  get_desc <- function(actual, predicted) {
    sp_rho <- NA
    sp_p   <- NA
    tryCatch({
      sp_rho <- cor(
        actual, predicted, method = "spearman", use = "complete.obs"
      )
      sp_p  <- round(
        cor.test(actual, predicted, method = "spearman")$p.value,
        4
      )
    }, error = function(e) {}
    )
    rmse <- sqrt(mean((actual - predicted)^2))
    return(c(sp_rho, sp_p, rmse))
  }
}


## Simulate
{
  print(here())
  if (run_bay) {
    bayes_model <- stan_model(here("models", "bayesian-ind.stan"))
  }
  mle_model   <- stan_model(here("models", "individual-uniform.stan"))

  tic()

  sim_counter <- 1
  for (t in seq_along(n_trials)){
    for (p in seq_along(n_players)){
      for (s in 1:n_sims){
        alpha <- center(rnorm(n_players[p]))
        df    <- simulate_data(n_players[p], n_trials[t], alpha)
        model_data <- list(
          K = n_players[p],
          N = n_trials[t],
          player0 = df$player0,
          player1 = df$player1,
          y = df$winner
        )
        mle_est <- optimizing(mle_model, data = model_data)
        if (run_bay) {
          ind_post <- sampling(bayes_model, data = model_data)
        }
        alpha_star  <- mle_est$par[paste("alpha[", 1:n_players[p], "]", sep = "")]
        if (run_bay) {
          alpha_hat   <- rep(NA, n_players[p])
          for (player in 1:n_players[p]){
            alpha_hat[player] <- mean(extract(ind_post)$alpha[, player])
          }
          ranked_bay <- extract(ind_post)$ranking
        }

        ranked_mle  <- mle_est$par[paste("ranked[", 1:n_players[p], "]",sep = "")]
        rank_actual <- abs(rank(alpha) - n_players[p]) + 1 # high to low
        rank_pred   <- abs(rank(mle_est$par[paste("alpha[", 1:n_players[p], "]",sep = "")]) - n_players[p])

        gr <- graph(edges = get_edges(df), n = n_players[p])
        sim_data[sim_counter, ] <- c(
          ceiling(n_trials[t] / p_trials * scaling),
          n_trials[t] * scaling,
          n_players[p] * scaling,
          s,
          get_desc(rank_actual, rank_pred),
          get_graph_desc(gr)
        )
        sim_counter <- sim_counter + 1
        print(sim_counter / n_sims)
      }
    }
  }
  toc()

  if (save_data){
    write.csv(
      sim_data,
      paste0(
        min_p, "-", max_p, "_images", min_players, "-", max_players, ".csv"),
        row.names = FALSE
    )
  }

  View(sim_data)

}
