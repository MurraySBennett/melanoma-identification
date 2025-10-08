## load libraries ----
{
  rm(list = ls())
  library(tidyverse)
  library(here)
  library(grtools)
  library(gridExtra)
  library(ggplotify)
}


## Plotting functions ----

get_label_pos <- function(usr) {
  positions = list(
    top_left = usr[1] + 0.25 * (usr[2] - usr[1]),
    top_right = usr[1] + 0.75 * (usr[2] - usr[1]),
    top_buffer = usr[4] + 0.5,
    
    left_top = usr[3] + 0.75 * (usr[4] - usr[3]),
    left_bottom = usr[3] + 0.25 * (usr[4] - usr[3]),
    left_buffer = usr[1] - 0.5,
  )
  return(positions)
}

plot.grt_wind_fit <- function(
  model, labels = c("dim A", "dim B"),
  assumptions = NA, pID = -1,
  ellipse_width = 0.8
  ) {
  if (dev.cur() > 1) {
    dev.off()
    dev.new()
  }
  on.exit(dev.off())
  plot.new()

  par(
    mar = c(5, 5, 5, 5) + 0.1,
    fig = c(0, 1, 0, 1),
    lty = 1, new = TRUE
  )
  buffer <- 2
  ranx <- c(min(model$means[, 1] - buffer), max(model$means[, 1]) + buffer)
  rany <- c(min(model$means[, 2] - buffer), max(model$means[, 2]) + buffer)

  # draw model$means of distributions
  p <- as.grob(function() {
    plot(
      model$means[, 1], model$means[, 2],
      pch = 3, xlim = ranx, ylim = rany,
      xlab = "", ylab = "",
      xaxt = "n", yaxt = "n",
      cex.lab = 1,
      font = 2
    )
    
    ## dimension A: top
    mtext(
      labels[1],
      side = 3, outer = FALSE, line = 1, 
      cex = 1.5, font = 2
    )
    mtext(
      "Low", side = 3, outer = FALSE, line = 0, at = -0.5, 
      cex = 1, font = 1
    )
    mtext(
      "High", side = 3, outer = FALSE, line = 0, at = 2, 
      cex = 1, font = 1
    )
    
    ## dimension B: side
    mtext(
      labels[2],
      side = 2, outer = FALSE, line = 1, 
      cex = 1.5, font = 2
    )
    mtext(
      "Low", side = 2, outer = FALSE, line = 0, at = -0.5, 
      cex = 1, font = 1
    )
    mtext(
      "High", side = 2, outer = FALSE, line = 0, at = 2, 
      cex = 1, font = 1
    )
    
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
  })
  return(p)
}



plot_ind_grt <- function(
    model, pID, assumptions = NA, ellipse_width = 0.8
) {
  model <- model$best_model
  
  buffer <- 1.5 # 2
  ranx <- c(min(model$means[, 1] - buffer), max(model$means[, 1]) + buffer)
  rany <- c(min(model$means[, 2] - buffer), max(model$means[, 2]) + buffer)
  
  if (dev.cur() > 1) {
    dev.off()
    dev.new()
  }
  on.exit(dev.off())
  plot.new()
  
  par(
    mar = c(0.5, 0.5, 0.5, 0.5),
    oma = c(0.5, 0.5, 0.5, 0.5)
  )
  
  
  p <- as.grob(function() {
    plot(
      model$means[, 1], model$means[, 2],
      pch = 3, xlim = ranx, ylim = rany,
      xlab = "", ylab = "",
      xaxt = "n", yaxt = "n"
    )
    if (!is.na(assumptions)) {
      text(
        x = ranx[1], y = rany[2],  # top-left
        labels = assumptions,
        adj = c(0, 1),  # Left-top alignment
        cex = 0.8,      # text size
        font = 2
      )
    }
    text(
      x = ranx[2], y = rany[1],  # bottom-right
      labels = paste0("p", pID),
      adj = c(1, 0),  # Left-top alignment
      cex = 0.8,      # text size
      # font = 2        # bold text
    )
    
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
      z <- matrix(0, nrow=n, ncol=n)
      for (i in 1:n){
        for (j in 1:n) {
          z[i,j] <- ellipse(x[i], y[j])
        }
      }
      contour(x, y, matrix(z, n, n),
              levels = ellipse_width, drawlabels = FALSE, add = TRUE
      )
    }
  })
  return(p)
}



get_assumptions <- function(held_assumptions, labels, n) {
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
  title <- NA
  return(title)
}

# Assign individual plots to specific positions
get_grid <- function(max_plots){
  grid_layout <- matrix(NA, nrow = 6, ncol = 4)
  grid_layout[1, ] <- seq(1,4)
  grid_layout[2, ] <- seq(5,8)
  grid_layout[3:4, 1] <- c(9,11)
  grid_layout[3:4, 4] <- c(10,12)
  grid_layout[5, ] <- seq(13,16)
  grid_layout[6, 1:4-(20-max_plots)] <- seq(17, max_plots)
  grid_layout[3:4, 2:3] <- max(grid_layout, na.rm = T) + 1
  return(grid_layout)
}

# read models ----
{
  ab_models <- readRDS(here("grt", "model_outputs", "ab_models.rds"))
  ac_models <- readRDS(here("grt", "model_outputs", "ac_models.rds"))
  bc_models <- readRDS(here("grt", "model_outputs", "bc_models.rds"))

  wind_ab <- readRDS(here("grt", "model_outputs", "ab_model_wind.rds"))
  wind_ac <- readRDS(here("grt", "model_outputs", "ac_model_wind.rds"))
  wind_bc <- readRDS(here("grt", "model_outputs", "bc_model_wind.rds"))
}


# Individual plots ----

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
group_models <- list(wind_ab, wind_ac, wind_bc)
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

  
  individual_plots <- lapply(
    seq(1, max(d$pID)), function(i) {
      curr_id <- d$pID[i]
      
      held_assumptions <- c(d$PS_A[i], d$PS_B[i], d$PI[i])
      assumptions <- get_assumptions(
        held_assumptions,
        labels = c(short_labels[cond_counter, 1], short_labels[cond_counter, 2])
      )
      
      plot_ind_grt(
        model = models[[curr_id]],
        pID = curr_id,
        assumptions = assumptions
      )
    }
  )
  labels = c(condition_labels[cond_counter, 1], condition_labels[cond_counter, 2])
  group_plot <- plot(group_models[[cond_counter]], labels = labels)
  plots <- c(individual_plots, list(group_plot))
  final_plot <- arrangeGrob(
    grobs = plots,
    layout_matrix = get_grid(max(d$pID)),
    top = NULL,
    padding = unit(0, "lines")
  )
  pdf(
    here("grt", "figures", "grt_models_ind", paste0(c, "_grouped_all.pdf")),
    width = 8, height = 11 #paper = "a4"
  )
  grid::grid.draw(final_plot)
  dev.off()
  
  cond_counter <- cond_counter + 1
}

