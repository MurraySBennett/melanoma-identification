data {
  int<lower = 0> K;
  int<lower = 0> N;
  int<lower = 1, upper = K> player1[N];
  int<lower = 1, upper = K> player0[N];
  int<lower = 0, upper = 1> y[N];
}
parameters {
  vector[K] alpha; // ability for player n
}
model {
  alpha ~ normal(0, 1);
  y ~ bernoulli_logit(alpha[player1] - alpha[player0]);
}
generated quantities {
  int<lower=1, upper=K> ranking[K];       // rank of player ability
  {
    int ranked_index[K] = sort_indices_desc(alpha);
    for (k in 1:K)
      ranking[ranked_index[k]] = k;
  }
}
