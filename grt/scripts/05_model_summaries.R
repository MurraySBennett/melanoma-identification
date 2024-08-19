rm(list = ls())
library(here)
library(tidyverse)
library(stringr)

save_path <- here("grt", "model_outputs")
file_list <- list.files(
    path = save_path, pattern = "*.txt", full.names = TRUE
)

get_lines <- function(file_path, lines_to_keep) {
    content <- readLines(file_path)
    if (grepl("fail", content[1])) {
        lines_to_keep[-1] <- lines_to_keep[-1] + 2
    }
    save_content <- trimws(content[lines_to_keep])
    file_id <- sub(
        "\\.txt$", "", basename(file_path)
    )
    save_content <- c(file_id, save_content, "")
    return(save_content)
}

lines_to_df <- function(file_path, lines_to_keep) {
    return_data <- list()
    file_id <- strsplit(
        sub(
            "\\.txt$", "", basename(file_path)
        ),
        "_"
    )
    condition <- file_id[1]
    pID <- file_id[2]
    return_data <- c(return_data, pID, condition)

    content <- readLines(file_path)
    if (grepl("fail", content[1])) {
        lines_to_keep[-1] <- lines_to_keep[-1] + 2
        return_data <- c(return_data, "maybe failed")
    } else {
        return_data <- c(return_data, "success")
    }
    best_model <- content[lines_to_keep[2]]
    best_model <- unlist(strsplit(
        trimws(unlist(strsplit(best_model, "\\{"))[2]),
        "\\}"))[1]
    PS_A <- trimws(unlist(strsplit(content[lines_to_keep[3]], ":"))[2])
    PS_B <- trimws(unlist(strsplit(content[lines_to_keep[4]], ":"))[2])
    PI <- trimws(unlist(strsplit(content[lines_to_keep[5]], ":"))[2])
    return_data <- c(return_data, best_model, PS_A, PS_B, PI)
    return(unlist(return_data))
}

#lines
# 1 - optimisation success
# 2 - best model AIC
# 20-23: assumption outcomes
lines_to_keep <- c(1, 6, 7, seq(20, 23))
all_lines <- list()
check_file_terms = c("summarised", "group", "LR")
for (file in file_list) {
    if (grepl(paste(check_file_terms, collapse = "|"), file)) {
        next
    }
    save_lines <- get_lines(file, lines_to_keep)
    all_lines <- c(all_lines, save_lines)
}
all_lines <- unlist(all_lines)
writeLines(all_lines, con=here(save_path, "00_summarised.txt"))

summary_data <- data.frame(
    condition = character(),
    pID = integer(),
    optim_result = logical(),
    best_model = character(),
    PS_A = character(),
    PS_B = character(),
    PI = character()
)
summary_lines <- c(1, seq(20,23))
counter <- 1
for (file in file_list) {
    if (grepl(paste(check_file_terms, collapse = "|"), file)) {
        next
    }
    summary_data[counter, ] <- lines_to_df(file, summary_lines)
    counter <- counter + 1
}

summary_data$pID <- as.integer(summary_data$pID)
summary_data <- summary_data %>% arrange(
    condition,
    desc(PI), desc(PS_A), desc(PS_B),
    pID
    )

write.csv(
    summary_data,
    here(save_path, "00_summary_table.csv"),
    row.names = FALSE
)

# meaningless table - completely misses the point of individual differences.
trends <- summary_data %>%
    group_by(condition) %>%
    filter(! is.na(pID)) %>%
    summarise(
        p    = n(),
        PS_A = sum(PS_A == "yes"),
        PS_B = sum(PS_B == "yes"),
        PI   = sum(PI == "yes")
    )
print(trends)
