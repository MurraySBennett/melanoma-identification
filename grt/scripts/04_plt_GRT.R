## load libraries ----
{
  rm(list = ls())
  library(tidyverse)
  library(here)
  library(grtools)
  library(gridExtra)
}


## Plotting functions ----
# if you want to be plotting the group, you need to change this to plot.grt_wind_fit
plot.grt_hm_fit <- function(
  model, labels = c("dim A", "dim B"),
  marginals = TRUE, scatter = TRUE,
  show_assumptions = FALSE, show_labels = TRUE, pID = -1,
  ellipse_width = 0.8
  ) {
  if (dev.cur() > 1) {
    dev.off()
    dev.new()
  }
  on.exit(dev.off())
  
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
    xlab = ifelse(show_labels, labels[1], ""),
    ylab = ifelse(show_labels, labels[2], ""),
    cex.lab = ifelse(marginals, 1.5, 2)
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


subplot_grt <- function(
  model, pID, title = NA, show_title = TRUE, ellipse_width = 0.8
  ) {
  model <- model$best_model

  buffer <- 1.5 # 2
  ranx <- c(min(model$means[, 1] - buffer), max(model$means[, 1]) + buffer)
  rany <- c(min(model$means[, 2] - buffer), max(model$means[, 2]) + buffer)

  plot(
    model$means[, 1], model$means[, 2],
    pch = 3, xlim = ranx, ylim = rany,
    xlab = "", ylab = "",
    xaxt = "n", yaxt = "n"
  )
  mtext(paste("Participant ID:", pID), side = 3, line = 0, outer = FALSE)
  if (show_title) {
    mtext(title, side = 3, line = 1, outer = FALSE)
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
}


get_assumption_title <- function(held_assumptions, labels, n) {
  if (any(held_assumptions)) {
    title <- c()
    assumptions <- c(
      paste0("PS(", labels[1], ")"),
      paste0("PS(", labels[2], ")"),
      "PI"
    )
    counter <- 1
    for (held in held_assumptions) {
      if (held) {
        held_assumptions[counter] <- FALSE
        title <- paste0(
          title,
          assumptions[counter]
        )
        if (any(held_assumptions)) {
          title <- paste0(title, " | ")
        }
      }
      counter <- counter + 1
    }
    return(title)
  }
  title <- "None"
  return(title)
}

# read models ----
{
  our_models <- readRDS(here("grt", "model_outputs", "JWH_MSB_models.rds"))

  ab_models <- readRDS(here("grt", "model_outputs", "ab_models.rds"))
  ac_models <- readRDS(here("grt", "model_outputs", "ac_models.rds"))
  bc_models <- readRDS(here("grt", "model_outputs", "bc_models.rds"))

  wind_ab <- readRDS(here("grt", "model_outputs", "ab_model_wind.rds"))
  wind_ac <- readRDS(here("grt", "model_outputs", "ac_model_wind.rds"))
  wind_bc <- readRDS(here("grt", "model_outputs", "bc_model_wind.rds"))
}

{ # Joe/Murray data
  counter <- 1
  our_labels <- c("JWH", "MSB")
  for (m in our_models) {
    pdf(
      here("grt", "figures", "grt_models_ind",
          paste0("ab_", our_labels[counter], ".pdf"
        )
      ),
      width = 6, height = 6
    )
    plot(m, labels = c("Shape Symmetry", "Border Regularity"))
    dev.off()
    counter <- counter + 1
  }
}

# Individual plots ----
{
  counter <- 1
  for (m in ab_models) {
    pdf(
      here(
        "grt", "figures", "grt_models_ind", paste0("ab_", counter, ".pdf")
      ),
      width = 6, height = 6
    )
    plot(
      model = m,
      labels = NA, #c("Symmetry", "Border Regularity"),
      marginals = FALSE,
      scatter = FALSE,
      show_assumptions = TRUE,
      show_labels = FALSE,
      pID = counter
    )
    dev.off()
    counter <- counter + 1
  }
}

{
  ## Shape Symmetry x Border Regularity
  counter <- 1
  for (m in ac_models) {
    pdf(
      here(
        "grt", "figures", "grt_models_ind", paste0("ac_", counter, ".pdf")
      ),
      width = 6, height = 6
    )
    plot(
      model = m,
      labels = NA, #c("Symmetry", "Colour Uniformity"),
      marginals = FALSE,
      scatter = FALSE,
      show_assumptions = TRUE,
      show_labels = FALSE,
      pID = counter
    )
    dev.off()
    counter <- counter + 1
  }
}

{
  ## Border Regularity x Colour Uniformity
  counter <- 1
  for (m in bc_models) {
    pdf(
      here(
        "grt", "figures", "grt_models_ind", paste0("bc_", counter, ".pdf")
      ),
      width = 6, height = 6
    )
    plot(
      model = m,
      labels = NA, #c("Border Regularity", "Colour Uniformity"),
      marginals = FALSE,
      scatter = FALSE,
      show_assumptions = TRUE,
      show_labels = FALSE,
      pID = counter
    )
    dev.off()
    counter <- counter + 1
  }
}

# Group model plots ----
{
  # Using the plotting function from the IAT_SFT project to remove marginals and add labels
  cat_levels=c("Low", "High", "Low", "High")
  # Symmetry x Border
  pdf(
    here(
      "grt", "figures", "grt_models_group", "ab_wind.pdf"
    ),
    width = 6, height = 6
  )
  plot(wind_ab, cat_labels = c("Shape Symmetry", "Border Regularity"), cat_levels=cat_levels)
  # plot(
  #   model = wind_ab,
  #   labels = c("Shape Symmetry", "Border Regularity"),
  #   show_assumptions = FALSE,
  #   marginals = TRUE,
  #   scatter = TRUE
  # )
  dev.off()

  
  # Symmetry x Colour
  pdf(
    here(
      "grt", "figures", "grt_models_group", "ac_wind.pdf"
    ),
    width = 6, height = 6
  )
  # plot(
  #   model = wind_ac,
  #   labels = c("Shape Symmetry", "Colour Uniformity"),
  #   show_assumptions = FALSE,
  #   marginals = TRUE,
  #   scatter = TRUE
  # ) 
  plot(wind_ac, cat_labels = c("Shape Symmetry", "Colour Uniformity"), cat_levels=cat_levels)
  dev.off()

  # Border x Colour
  pdf(
    here(
      "grt", "figures", "grt_models_group", "bc_wind.pdf"
    ),
    width = 6, height = 6
  )
  # plot(
  #   model = wind_bc,
  #   labels = c("Border Regularity", "Colour Uniformity"),
  #   show_assumptions = FALSE,
  #   marginals = TRUE,
  #   scatter = TRUE
  # )
  plot(wind_bc, cat_labels = c("Border Regularity", "Colour Uniformity"), cat_levels=cat_levels)
  dev.off()
}

## All individual plots - Appendices ----
## return plot objects to a sorted list: group plots by assumptions
figures <- data.frame(
  pID = NULL,
  cond = NULL,
  PS_A = NULL,
  PS_B = NULL,
  PI = NULL
)

cond_counter <- 1
model_list <- list(ab_models, ac_models, bc_models)
conds <- c("ab", "ac", "bc")
for (models in model_list) {
  counter <- 1
  for (m in models) {
    figures <- rbind(figures,
      list(
        pID = counter,
        cond = conds[cond_counter],
        PS_A = ifelse(grepl("PS\\(A\\)", m$best_model$model), TRUE, FALSE),
        PS_B = ifelse(grepl("PS\\(B\\)", m$best_model$model), TRUE, FALSE),
        PI = ifelse(grepl("PI", m$best_model$model), TRUE, FALSE)
      )
    )
    counter <- counter + 1
  }
  cond_counter <- cond_counter + 1
}

figures <- figures %>%
  arrange(cond, desc(PS_A), desc(PS_B), desc(PI)) %>%
  group_by(cond, PS_A, PS_B, PI) %>%
  mutate(
    n = n()
  )

condition_labels <- matrix(
  c(
    "Shape Symmetry",     "Shape Symmetry",     "Border Regularity",
    "Border Regularity",  "Colour Uniformity",  "Colour Uniformity"
  ),
  nrow = 3, ncol = 2
)
short_labels <- matrix(
  c(
    "Shape", "Shape", "Border",
    "Border", "Colour", "Colour"
  ),
  nrow = 3, ncol = 2
)

cond_counter <- 1
for (c in conds) {
  d <- figures[figures$cond == c, ]
  models <- model_list[[cond_counter]]

  held_assumptions <- c( d$PS_A[1], d$PS_B[1], d$PI[1])
  curr_title <- get_assumption_title(
    held_assumptions,
    labels = c(short_labels[cond_counter, 1], short_labels[cond_counter, 2])
  )
  pdf(
    here("grt", "figures", "grt_models_ind", paste0(c, "_ind_all.pdf")),
    width = 8, height = 11 #paper = "a4"
  )
  par(
    mfrow = c(5, 4),
    oma = c(2, 3, 2, 0),
    mar = c(2, 3, 2, 2)
  )
  for (plt in 1:max(d$pID)) {
    curr_id <- d$pID[plt]
    held_assumptions <- c(d$PS_A[plt], d$PS_B[plt], d$PI[plt])
    new_title <- get_assumption_title(
      held_assumptions,
      labels = c(short_labels[cond_counter, 1], short_labels[cond_counter, 2])
    )
    if (curr_title == new_title && plt > 1) {
      show_title <- FALSE
    } else {
      curr_title <- new_title
      show_title <- TRUE
    }
    print(curr_title)
    subplot_grt(
      model = models[[curr_id]],
      pID = curr_id,
      title = ifelse(
        d$n[plt] > 1, paste(curr_title, expression("\u2192")), curr_title
      ),
      show_title = show_title
    )
  }
  mtext(
    condition_labels[cond_counter, 1],
    side = 1, outer = TRUE, line = 0, cex = 3
  )
  mtext(
    condition_labels[cond_counter, 2],
    side = 2, outer = TRUE, line = 0, cex = 3
  )
  dev.off()
  cond_counter <- cond_counter + 1
}

