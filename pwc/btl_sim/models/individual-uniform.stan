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
  y ~ bernoulli_logit(alpha[player1] - alpha[player0]);
}
generated quantities {
  int<lower=1, upper=K> ranked[K] = sort_indices_desc(alpha);
}
