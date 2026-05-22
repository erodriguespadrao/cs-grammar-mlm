
# Required packages:
#   install.packages(c("tidyverse", "ggplot2", "effsize", "coin",
#                      "ggpubr", "scales", "patchwork"))
#
# Usage:
#   Rscript analyse_results.R
#
# Input:  pairs_scored.csv  (in working directory)
# Output: results/ folder containing:
#           boxplot_trimmed.png/.pdf
#           boxplot_full.png/.pdf
#           accuracy_by_violation.png/.pdf
#           results_summary.csv
# ------------

library(tidyverse)
library(ggplot2)
library(effsize)   # Cohen's d
library(coin)      # Wilcoxon signed-rank (exact)
library(patchwork) # combine plots
library(scales)    # comma formatting

# ── Configuration

OUTLIER_THRESHOLD <- 5000

# Uncomment for Salazar results
# OUTPUT_DIR        <- "results_salazar"
# INPUT_FILE        <- "pairs_scored_salazar.csv"

# Uncomment for Kauf
OUTPUT_DIR        <- "results_kauf"
INPUT_FILE        <- "pairs_scored_kauf.csv"

dir.create(OUTPUT_DIR, showWarnings = FALSE)

# Colour palette
COL_VALID   <- "#2196F3"
COL_INVALID <- "#F44336"
COL_MBERT   <- "#4CAF50"
COL_XLMR    <- "#FF9800"

# ggplot theme
theme_thesis <- function() {
  theme_classic() +
  theme(
    text             = element_text(family = "sans", size = 11),
    plot.title       = element_text(size = 13, face = "bold", hjust = 0.5),
    plot.subtitle    = element_text(size = 10, hjust = 0.5, color = "grey40"),
    axis.title       = element_text(size = 12),
    axis.text        = element_text(size = 10),
    legend.position  = "bottom",
    legend.title     = element_blank(),
    strip.text       = element_text(size = 11, face = "bold"),
    panel.grid.major.y = element_line(color = "grey90", linewidth = 0.4),
  )
}

# ── Load data

cat("Loading", INPUT_FILE, "...\n")
df <- read_csv(INPUT_FILE, show_col_types = FALSE)

cat("  Rows:", nrow(df), "\n")
cat("  Valid:", sum(df$label == 1), "\n")
cat("  Invalid:", sum(df$label == 0), "\n\n")

# Separate valid and invalid
valid_df   <- df %>% filter(label == 1)
invalid_df <- df %>% filter(label == 0)

# Pair-level summary: for each pair_id, compute
#   valid PPPL and mean of invalid PPPLs
pair_summary <- df %>%
  group_by(pair_id) %>%
  summarise(
    valid_mbert   = mbert_pppl[label == 1],
    valid_xlmr    = xlmr_pppl[label == 1],
    invalid_mbert = mean(mbert_pppl[label == 0]),
    invalid_xlmr  = mean(xlmr_pppl[label == 0]),
    diff_mbert    = valid_mbert - invalid_mbert,
    diff_xlmr     = valid_xlmr  - invalid_xlmr,
    .groups = "drop"
  ) %>%
  filter(!is.na(valid_mbert), !is.na(invalid_mbert))

cat("Pairs with both valid and invalid:", nrow(pair_summary), "\n\n")


# ── Statistical tests

run_stats <- function(pair_sum, model_label, diff_col,
                      valid_col, invalid_col,
                      valid_all, invalid_all) {

  diffs   <- pair_sum[[diff_col]]
  v_all   <- valid_all[[if (model_label == "mBERT") "mbert_pppl" else "xlmr_pppl"]]
  i_all   <- invalid_all[[if (model_label == "mBERT") "mbert_pppl" else "xlmr_pppl"]]

  # Paired t-test 
  t_res  <- t.test(diffs, mu = 0, alternative = "less")

  # Wilcoxon signed-rank test 
  w_res  <- wilcox.test(diffs, mu = 0, alternative = "less", exact = FALSE)

  # Cohen's d
  d_res  <- cohen.d(v_all, i_all)

  # Pair-level accuracy
  correct <- sum(diffs < 0)
  total   <- length(diffs)

  cat("──", model_label, "──\n")
  cat(sprintf("  Valid   mean PPPL : %.2f  (median %.2f)\n",
              mean(v_all), median(v_all)))
  cat(sprintf("  Invalid mean PPPL : %.2f  (median %.2f)\n",
              mean(i_all), median(i_all)))
  cat(sprintf("  Paired t-test     : t(%.0f) = %.3f,  p = %.4f%s\n",
              t_res$parameter, t_res$statistic, t_res$p.value,
              ifelse(t_res$p.value < .001, "  ***",
              ifelse(t_res$p.value < .01,  "  **",
              ifelse(t_res$p.value < .05,  "  *", "")))))
  cat(sprintf("  Wilcoxon          : W = %.1f,  p = %.4f%s\n",
              w_res$statistic, w_res$p.value,
              ifelse(w_res$p.value < .001, "  ***",
              ifelse(w_res$p.value < .01,  "  **",
              ifelse(w_res$p.value < .05,  "  *", "")))))
  cat(sprintf("  Cohen's d         : %.4f  [95%% CI: %.4f, %.4f]\n",
              d_res$estimate, d_res$conf.int[1], d_res$conf.int[2]))
  cat(sprintf("  Pair accuracy     : %d/%d = %.1f%%\n\n",
              correct, total, correct / total * 100))

  list(
    model            = model_label,
    valid_mean       = mean(v_all),
    valid_median     = median(v_all),
    invalid_mean     = mean(i_all),
    invalid_median   = median(i_all),
    t_statistic      = t_res$statistic,
    t_df             = t_res$parameter,
    t_p              = t_res$p.value,
    wilcoxon_W       = w_res$statistic,
    wilcoxon_p       = w_res$p.value,
    cohens_d         = d_res$estimate,
    cohens_d_ci_low  = d_res$conf.int[1],
    cohens_d_ci_high = d_res$conf.int[2],
    pair_accuracy    = correct / total * 100,
    pairs_correct    = correct,
    pairs_total      = total
  )
}

cat("=" %>% strrep(60), "\n")
cat("RESULTS REPORT\n")
cat("=" %>% strrep(60), "\n\n")

mb_stats <- run_stats(pair_summary, "mBERT", "diff_mbert",
                      "valid_mbert", "invalid_mbert",
                      valid_df, invalid_df)
xl_stats <- run_stats(pair_summary, "XLM-R", "diff_xlmr",
                      "valid_xlmr", "invalid_xlmr",
                      valid_df, invalid_df)


# ── Accuracy by violation type

# For each invalid sentence, check whether valid PPPL < that invalid's PPPL
vtype_acc <- df %>%
  filter(label == 0) %>%
  left_join(
    valid_df %>% select(pair_id, valid_mbert = mbert_pppl, valid_xlmr = xlmr_pppl),
    by = "pair_id"
  ) %>%
  mutate(
    correct_mbert = valid_mbert < mbert_pppl,
    correct_xlmr  = valid_xlmr  < xlmr_pppl
  ) %>%
  group_by(violation_type) %>%
  summarise(
    n             = n(),
    acc_mbert     = mean(correct_mbert) * 100,
    acc_xlmr      = mean(correct_xlmr)  * 100,
    .groups = "drop"
  ) %>%
  filter(n >= 20) %>%
  arrange(desc(acc_mbert))

# Shorten violation type labels
short_labels <- c(
  "DET-NOUN dependency"    = "DET-NOUN",
  "DET-PROPN dependency"   = "DET-PROPN",
  "DET-ADJ (pre-nominal)"  = "DET-ADJ",
  "ADJ-NOUN dependency"    = "ADJ-NOUN",
  "ADJ-PROPN dependency"   = "ADJ-PROPN",
  "AUX-VERB dependency"    = "AUX-VERB",
  "AUX-AUX (modal+have)"   = "AUX-AUX",
  "ADP-DET (prep phrase)"  = "ADP-DET",
  "ADP-NOUN (prep phrase)" = "ADP-NOUN",
  "ADP-PROPN (prep phrase)"= "ADP-PROPN",
  "ADP-PRON (prep phrase)" = "ADP-PRON",
  "PART-VERB dependency"   = "PART-VERB",
  "NUM-NOUN dependency"    = "NUM-NOUN"
)

vtype_acc <- vtype_acc %>%
  mutate(short = coalesce(short_labels[violation_type], violation_type),
         label_n = paste0(short, "\n(n=", n, ")"))

cat("Accuracy by violation type (mBERT):\n")
vtype_acc %>%
  select(violation_type, n, acc_mbert, acc_xlmr) %>%
  print(n = 20)
cat("\n")


# ── Save results summary CSV

summary_df <- bind_rows(
  as_tibble(mb_stats),
  as_tibble(xl_stats)
)
write_csv(summary_df, file.path(OUTPUT_DIR, "results_summary.csv"))
cat("Saved:", file.path(OUTPUT_DIR, "results_summary.csv"), "\n")


# ── save plot as both PNG and PDF

save_plot <- function(p, stem, w = 11, h = 5) {
  ggsave(file.path(OUTPUT_DIR, paste0(stem, ".png")),
         plot = p, width = w, height = h, dpi = 150)
  ggsave(file.path(OUTPUT_DIR, paste0(stem, ".pdf")),
         plot = p, width = w, height = h)
  cat("Saved:", file.path(OUTPUT_DIR, paste0(stem, ".png/.pdf")), "\n")
}


# ── Box plots

make_boxplot <- function(valid_d, invalid_d, subtitle = "") {

  plot_df <- bind_rows(
    valid_d   %>% select(mbert_pppl, xlmr_pppl) %>% mutate(condition = "Valid"),
    invalid_d %>% select(mbert_pppl, xlmr_pppl) %>% mutate(condition = "Invalid")
  ) %>%
  pivot_longer(c(mbert_pppl, xlmr_pppl),
               names_to = "model", values_to = "pppl") %>%
  mutate(
    model     = recode(model, mbert_pppl = "mBERT", xlmr_pppl = "XLM-RoBERTa"),
    condition = factor(condition, levels = c("Valid", "Invalid"))
  )

  # Compute medians for annotation
  medians <- plot_df %>%
    group_by(model, condition) %>%
    summarise(med = median(pppl), .groups = "drop")

  ggplot(plot_df, aes(x = condition, y = pppl, fill = condition)) +
    geom_boxplot(
      outlier.size  = 0.8,
      outlier.alpha = 0.3,
      width         = 0.5,
      linewidth     = 0.6
    ) +
    geom_text(
      data = medians,
      aes(x = condition, y = med, label = round(med, 0)),
      vjust = -0.6, size = 3, fontface = "bold", inherit.aes = FALSE
    ) +
    facet_wrap(~model, scales = "free_y") +
    scale_fill_manual(values = c(Valid = COL_VALID,
                                 Invalid = COL_INVALID)) +
    scale_y_continuous(labels = comma) +
    labs(
      title    = "Pseudo-Perplexity: Valid vs Invalid Code-Switched Sentences",
      subtitle = subtitle,
      x        = NULL,
      y        = "Pseudo-Perplexity (PPPL)"
    ) +
    theme_thesis() +
    theme(legend.position = "none")
}

# Trimmed version
n_removed <- sum(valid_df$mbert_pppl > OUTLIER_THRESHOLD |
                 valid_df$xlmr_pppl  > OUTLIER_THRESHOLD) +
             sum(invalid_df$mbert_pppl > OUTLIER_THRESHOLD |
                 invalid_df$xlmr_pppl  > OUTLIER_THRESHOLD)

p_trimmed <- make_boxplot(
  valid_df   %>% filter(mbert_pppl <= OUTLIER_THRESHOLD, xlmr_pppl <= OUTLIER_THRESHOLD),
  invalid_df %>% filter(mbert_pppl <= OUTLIER_THRESHOLD, xlmr_pppl <= OUTLIER_THRESHOLD),
  subtitle = paste0("Outliers > ", OUTLIER_THRESHOLD, " removed (", n_removed, " sentences)")
)
save_plot(p_trimmed, "boxplot_trimmed")

# Full version
p_full <- make_boxplot(
  valid_df, invalid_df,
  subtitle = "All data including outliers"
)
save_plot(p_full, "boxplot_full")


# ── Accuracy by violation type plot

vtype_long <- vtype_acc %>%
  pivot_longer(c(acc_mbert, acc_xlmr),
               names_to = "model", values_to = "accuracy") %>%
  mutate(
    model    = recode(model, acc_mbert = "mBERT", acc_xlmr = "XLM-RoBERTa"),
    label_n  = factor(label_n,
                      levels = vtype_acc$label_n[order(vtype_acc$acc_mbert,
                                                        decreasing = TRUE)])
  )

p_vtype <- ggplot(vtype_long,
                  aes(x = label_n, y = accuracy, fill = model)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.65) +
  geom_hline(yintercept = 50, linetype = "dashed",
             color = "grey50", linewidth = 0.8) +
  annotate("text", x = 0.5, y = 51.5, label = "Chance (50%)",
           hjust = 0, size = 3, color = "grey50") +
  geom_text(
    aes(label = paste0(round(accuracy, 0), "%")),
    position = position_dodge(width = 0.7),
    vjust = -0.4, size = 2.8
  ) +
  scale_fill_manual(values = c("mBERT" = COL_MBERT,
                               "XLM-RoBERTa" = COL_XLMR)) +
  scale_y_continuous(limits = c(0, 108),
                     breaks = seq(0, 100, 25),
                     labels = function(x) paste0(x, "%")) +
  labs(
    title = "Pair-level Accuracy by EC Violation Type",
    x     = NULL,
    y     = "Pair-level Accuracy (%)"
  ) +
  theme_thesis() +
  theme(
    axis.text.x     = element_text(size = 8.5),
    legend.position = "bottom"
  )

save_plot(p_vtype, "accuracy_by_violation", w = 13, h = 6)

cat("\nAll outputs written to:", OUTPUT_DIR, "/\n")
