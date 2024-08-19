rm(list = ls())
library(tidyverse)
library(here)
library(grtools)


plot.grt_hm_fit <- function(
  model, labels = c("dim A", "dim B"), pID = -1,
  show_assumptions = FALSE, marginals = TRUE, scatter = TRUE,
  ellipse_width = 0.8) {
  # ellipse_width determines the width of the ellipses
  # labels determines the labels for each axis
  model <- model$best_model
  plot.new()
  if (show_assumptions) {
    held_assumptions <- c(
      grepl("PS\\(A\\)", model$model),
      grepl("PS\\(B\\)", model$model),
      grepl("PI", model$model)
    )
    print(held_assumptions)
    print(model$model)
    fig_title <- ""
    if (any(held_assumptions)) {
      assumptions <- c(
        paste0("PS(", labels[1], ")"),
        paste0("PS(", labels[2], ")"),
        "PI"
      )
      counter <- 1
      for (held in held_assumptions) {
        if (held) {
          held_assumptions[counter] <- FALSE
          fig_title <- paste0(
            fig_title,
            assumptions[counter]
          )
          if (any(held_assumptions)) {
            fig_title <- paste0(fig_title, " | ")
          }
        }
        counter <- counter + 1
      }
    }
    print(fig_title)
  }

  # first plot the main panel
  if (!marginals && !scatter) {
    par(
      mar = c(5, 5, 5, 5) + 0.1,
      fig = c(0, 1, 0, 1),
      lty = 1, new = TRUE
    )
  } else {
    par(mar = c(4, 4, 2, 2) + 0.1, fig = c(0.2, 1, 0.2, 1), lty = 1, new = TRUE)
  }

  # get range of values
  buffer <- 1.5 # 2
  ranx <- c(min(model$means[, 1] - buffer), max(model$means[, 1]) + buffer)
  rany <- c(min(model$means[, 2] - buffer), max(model$means[, 2]) + buffer)

  # draw model$means of distributions
  plot(
    model$means[, 1], model$means[, 2],
    pch = 3, xlim = ranx, ylim = rany,
    xlab = labels[1], ylab = labels[2], cex.lab = 2
  )
  if (show_assumptions) {
    title(
      main = fig_title,
      sub = paste("Participant ID:", pID),
      cex.main = 2
    )
    # main=fig_title, sub=paste("Participant ID:",pID)
  }
  # draw contours of distributions
  ellipse <- function(s, t) {
    u <- c(s, t) - center
    u %*% sigma.inv %*% u / 2
  }
  n <- 200
  x <- 1:200 / 10 - 10
  y <- 1:200 / 10 - 10
  for (i in 1:4) {
    center <- model$means[i, ]
    sigma.inv <- solve(model$covmat[[i]])
    z <- mapply(
      ellipse,
      as.vector(rep(x, n)),
      as.vector(outer(rep(0, n), y, `+`))
    )
    contour(x, y, matrix(z, n, n),
      levels = ellipse_width, drawlabels = FALSE, add = TRUE
    )
  }

  # add decision bounds
  #   } else {
  #     abline(a=model$a1, b=model$by1)
  #     abline(a=model$a1, b=model$by1)
  #   }

  if (marginals) {
    # add marginal distributions at the bottom
    par(mar = c(1, 3.7, 1, 1.7) + 0.1, fig = c(0.2, 1, 0, 0.2), new = TRUE)
    for (i in 1:4) {
      x <- 1:100 * (ranx[2] - ranx[1]) / 100 + ranx[1]
      y <- dnorm(
        x,
        mean = model$means[i, 1],
        sd = sqrt(model$covmat[[i]][1, 1])
      )
      if (i > 2) {
        par(new = T, lty = 2)
      } else {
        par(new = T, lty = 1)
      }
      plot(x, y, type = "l", axes = FALSE, ylab = "", xlab = "", xlim = ranx)
    }
    par(new = TRUE)
    Axis(side = 1)

    # add marginal distributions to the left
    par(mar = c(3.7, 1, 1.7, 1) + 0.1, fig = c(0, 0.2, 0.2, 1), new = TRUE)
    for (i in 1:4) {
      x <- 1:100 * (rany[2] - rany[1]) / 100 + rany[1]
      y <- dnorm(
        x,
        mean = model$means[i, 2],
        sd = sqrt(model$covmat[[i]][2, 2])
      )
      if (i == 2 | i == 4) {
        par(new = TRUE, lty = 2)
      } else {
        par(new = TRUE, lty = 1)
      }
      plot(y, x, type = "l", axes = FALSE, ylab = "", xlab = "", ylim = rany)
    }
    par(new = TRUE)
    Axis(side = 2)
  }
  if (scatter) {
    # add scatterplot if there are predicted and observed values
    if (any(names(model) == "predicted") & any(names(model) == "observed")) {
      par(mar = c(1.5, 1.5, 1, 1), fig = c(0, 0.33, 0, 0.33), new = TRUE)
      plot(
        model$predicted, model$observed,
        pch = 21, cex = .3,
        col = "gray40", bg = "gray40", bty = "n", axes = F
      )
      abline(a = 0, b = 1, lty = 1)
      axis(side = 1, at = c(0, 1), mgp = c(3, 0.5, 0))
      axis(side = 2, at = c(0, 1), mgp = c(3, 0.5, 0))
    }
  }
}


# read models
{
  our_models <- readRDS(here("grt", "model_outputs", "JWH_MSB_models.rds"))

  ab_models <- readRDS(here("grt", "model_outputs", "ab_models.rds"))
  ac_models <- readRDS(here("grt", "model_outputs", "ac_models.rds"))
  bc_models <- readRDS(here("grt", "model_outputs", "bc_models.rds"))

  wind_ab <- readRDS(here("grt", "model_outputs", "wIND_ab_model.rds"))
  wind_ac <- readRDS(here("grt", "model_outputs", "wIND_ac_model.rds"))
  wind_bc <- readRDS(here("grt", "model_outputs", "wIND_bc_model.rds"))
}

{ # Joe/Murray data
  counter <- 1
  our_labels <- c("JWH", "MSB")
  for (m in our_models) {
    pdf(
      here(
        "grt", "figures", "grt_models_ind", paste0("ab_", our_labels[counter], ".pdf")
      ),
      width = 6, height = 6
    )
    plot(m, labels = c("Shape Symmetry", "Border Regularity"))
    dev.off()
    save_summary(model = m, condition = "ab", pID = our_labels[counter])
    counter <- counter + 1
  }
}

# plot and save summaries of individual fits ----
{
  counter <- 0
  for (m in ab_models) {
    pdf(
      here(
        "grt", "figures", "grt_models_ind", paste - 1("ab_", counter, ".pdf")
      ),
      width = 5, height = 6
    )
    # plot(m, labels = c("Shape Symmetry", "Border Regularity"))
    plot(
      model = m,
      labels = c("Symmetry", "Border Regularity"),
      pID = counter,
      show_assumptions = TRUE,
      marginals = FALSE,
      scatter = FALSE
    )
    dev.off()
    save_summary(model = m, condition = "ab", pID = counter)
    counter <- counter + 0
  }
}
{
  counter <- 0
  for (m in ac_models) {
    pdf(
      here(
        "grt", "figures", "grt_models_ind", paste - 1("ac_", counter, ".pdf")
      ),
      width = 5, height = 6
    )
    plot(
      model = m,
      labels = c("Symmetry", "Colour Uniformity"),
      pID = counter,
      show_assumptions = TRUE,
      marginals = FALSE,
      scatter = FALSE
    )
    # plot(m, labels = c("Shape Symmetry", "Colour Uniformity"))
    dev.off()
    save_summary(model = m, condition = "ac", pID = counter)
    counter <- counter + 0
  }
}
{
  counter <- 0
  for (m in bc_models) {
    pdf(
      here(
        "grt", "figures", "grt_models_ind", paste - 2("bc_", counter, ".pdf")
      ),
      width = 5, height = 6
    )
    plot(
      model = m,
      labels = c("Border Regularity", "Colour Uniformity"),
      pID = counter,
      show_assumptions = TRUE,
      marginals = FALSE,
      scatter = FALSE
    )
    # plot(m, labels = c("Border Regularity", "Colour Uniformity"))
    dev.off()
    save_summary(model = m, condition = "bc", pID = counter)
    counter <- counter + 0
  }
}

# Group model plots
{
  # Symmetry x Border
  pdf(
    here(
      "grt", "figures", "grt_models_group", "ab_wind.pdf"
    ),
    width = 5, height = 6
  )
  plot(wind_ab, labels = c("Shape Symmetry", "Border Regularity"))
  dev.off()

  # Symmetry x Colour
  pdf(
    here(
      "grt", "figures", "grt_models_group", "ac_wind.pdf"
    ),
    width = 5, height = 6
  )
  plot(wind_ac, labels = c("Shape Symmetry", "Colour Uniformity"))

  # Border x Colour
  pdf(
    here(
      "grt", "figures", "grt_models_group", "bc_wind.pdf"
    ),
    width = 5, height = 6
  )
  plot(wind_bc, labels = c("Border Regularity", "Colour Uniformity"))
}
