# Model structure:
#   log(PPPL) ~ label + (1 | speaker) + (1 | conversation)
#
#   Fixed effect:   label (Valid vs Invalid) 
#   Random effects: speaker + file
#
# Required packages:
#   install.packages(c("tidyverse", "lme4", "lmerTest", "ggplot2",
#                      "scales", "patchwork"))
#
# Input:  pairs_scored_enriched.csv  (pairs_scored.csv + speaker/file columns)
#         Generate with: python enrich_scored.py
# Output: results_mixed/ folder
# -----------

library(tidyverse)
library(lme4)
library(lmerTest)
library(ggplot2)
library(scales)

# Uncomment for Salazar results
# INPUT_FILE <- "pairs_scored_salazar_enriched.csv"
# OUTPUT_DIR <- "results_mixed_salazar"

# Uncomment for Kauf
INPUT_FILE <- "pairs_scored_kauf_enriched.csv"
OUTPUT_DIR <- "results_mixed_kauf"

dir.create(OUTPUT_DIR, showWarnings = FALSE)

# Colour palette
COL_VALID   <- "#2196F3"
COL_INVALID <- "#F44336"
COL_MBERT   <- "#4CAF50"
COL_XLMR    <- "#FF9800"

theme_thesis <- function() {
  theme_classic() +
  theme(
    text               = element_text(family = "sans", size = 11),
    plot.title         = element_text(size = 13, face = "bold", hjust = 0.5),
    plot.subtitle      = element_text(size = 10, hjust = 0.5, color = "grey40"),
    axis.title         = element_text(size = 12),
    axis.text          = element_text(size = 10),
    strip.text         = element_text(size = 11, face = "bold"),
    panel.grid.major.y = element_line(color = "grey90", linewidth = 0.4),
    legend.position    = "none",
  )
}

save_plot <- function(p, stem, w = 10, h = 5) {
  ggsave(file.path(OUTPUT_DIR, paste0(stem, ".png")), plot=p, width=w, height=h, dpi=150)
  ggsave(file.path(OUTPUT_DIR, paste0(stem, ".pdf")), plot=p, width=w, height=h)
  cat("Saved:", file.path(OUTPUT_DIR, paste0(stem, ".png/.pdf\n")))
}


# ── Load data 

cat("Loading", INPUT_FILE, "...\n")
df <- read_csv(INPUT_FILE, show_col_types = FALSE) %>%
  mutate(
    label     = factor(label, levels = c(1, 0),
                       labels = c("Valid", "Invalid")),
    log_mbert = log(mbert_pppl),
    log_xlmr  = log(xlmr_pppl),
    speaker   = factor(speaker),
    conversation = factor(conversation),
  )

cat(sprintf("  %d rows | %d valid | %d invalid\n",
            nrow(df), sum(df$label == "Valid"), sum(df$label == "Invalid")))
cat(sprintf("  %d unique speakers | %d unique conversations\n\n",
            n_distinct(df$speaker), n_distinct(df$conversation)))


# ── Normality check 

cat("── Normality check ──\n")

set.seed(42)
sample_idx <- sample(nrow(df), min(nrow(df), 4000))

sw_raw_mb <- shapiro.test(df$mbert_pppl[sample_idx])
sw_log_mb <- shapiro.test(df$log_mbert[sample_idx])
sw_raw_xl <- shapiro.test(df$xlmr_pppl[sample_idx])
sw_log_xl <- shapiro.test(df$log_xlmr[sample_idx])

cat(sprintf("Shapiro-Wilk (n=%d sample):\n", length(sample_idx)))
cat(sprintf("  mBERT raw  PPPL: W=%.4f, p=%.2e\n", sw_raw_mb$statistic, sw_raw_mb$p.value))
cat(sprintf("  mBERT log  PPPL: W=%.4f, p=%.2e\n", sw_log_mb$statistic, sw_log_mb$p.value))
cat(sprintf("  XLM-R raw  PPPL: W=%.4f, p=%.2e\n", sw_raw_xl$statistic, sw_raw_xl$p.value))
cat(sprintf("  XLM-R log  PPPL: W=%.4f, p=%.2e\n", sw_log_xl$statistic, sw_log_xl$p.value))
cat("\nNote: Shapiro-Wilk is very sensitive at large n. Inspect Q-Q plots visually.\n\n")

# Q-Q plots: 2x2 grid (raw vs log, mBERT vs XLM-R)
qq_data <- df %>%
  select(label, mbert_pppl, xlmr_pppl, log_mbert, log_xlmr) %>%
  pivot_longer(c(mbert_pppl, xlmr_pppl, log_mbert, log_xlmr),
               names_to = "metric", values_to = "value") %>%
  mutate(
    model = if_else(str_detect(metric, "mbert"), "mBERT", "XLM-RoBERTa"),
    scale = factor(
      if_else(str_detect(metric, "log"), "Log-transformed", "Raw"),
      levels = c("Raw", "Log-transformed")
    )
  )

p_qq <- ggplot(qq_data, aes(sample = value, color = label)) +
  stat_qq(size = 0.4, alpha = 0.4) +
  stat_qq_line(linewidth = 0.8) +
  scale_color_manual(values = c(Valid = COL_VALID, Invalid = COL_INVALID)) +
  facet_grid(scale ~ model, scales = "free") +
  labs(
    title    = "Q-Q Plots: Raw vs Log-Transformed PPPL",
    subtitle = "Points on the diagonal = normally distributed",
    x = "Theoretical quantiles",
    y = "Sample quantiles",
    color = NULL
  ) +
  theme_thesis() +
  theme(legend.position = "bottom")

save_plot(p_qq, "normality_check", w = 10, h = 8)


# ── Log-scale box plots

cat("── Log-scale box plots ──\n")

plot_df <- df %>%
  select(label, log_mbert, log_xlmr) %>%
  pivot_longer(c(log_mbert, log_xlmr),
               names_to = "model", values_to = "log_pppl") %>%
  mutate(model = recode(model, log_mbert = "mBERT", log_xlmr = "XLM-RoBERTa"))

medians_log <- plot_df %>%
  group_by(model, label) %>%
  summarise(med = median(log_pppl), .groups = "drop")

p_log_box <- ggplot(plot_df, aes(x = label, y = log_pppl, fill = label)) +
  geom_boxplot(outlier.size = 0.6, outlier.alpha = 0.2,
               width = 0.5, linewidth = 0.6) +
  geom_text(data = medians_log,
            aes(x = label, y = med, label = round(med, 2)),
            vjust = -0.5, size = 3, fontface = "bold", inherit.aes = FALSE) +
  facet_wrap(~model) +
  scale_fill_manual(values = c(Valid   = paste0(COL_VALID,  "99"),
                               Invalid = paste0(COL_INVALID, "99"))) +
  labs(
    title    = "Log-Transformed Pseudo-Perplexity: Valid vs Invalid",
    subtitle = "Log scale removes heavy-tail distortion; gap is now clearly visible",
    x = NULL,
    y = "log(PPPL)"
  ) +
  theme_thesis()

save_plot(p_log_box, "boxplot_log")


# ── Linear mixed effects models

cat("\n── Step 3: Mixed effects models ──\n")
cat("Formula: log(PPPL) ~ label + (1 | speaker) + (1 | conversation)\n")
cat("  Fixed effect:   label (Valid=reference, Invalid=test)\n")
cat("  Random effects: by-speaker and by-conversation intercepts\n\n")

m_mbert <- lmer(log_mbert ~ label + (1 | speaker) + (1 | conversation),
                data = df, REML = TRUE)

m_xlmr  <- lmer(log_xlmr  ~ label + (1 | speaker) + (1 | conversation),
                data = df, REML = TRUE)

# Print summaries to console
cat("\n", strrep("=", 60), "\n")
cat("mBERT MIXED MODEL\n")
cat(strrep("=", 60), "\n")
print(summary(m_mbert))

cat("\n", strrep("=", 60), "\n")
cat("XLM-R MIXED MODEL\n")
cat(strrep("=", 60), "\n")
print(summary(m_xlmr))

# Save full summaries to text file
sink(file.path(OUTPUT_DIR, "mixed_model_summary.txt"))
cat(strrep("=", 60), "\n")
cat("mBERT MIXED MODEL\n")
cat("Formula: log(mbert_pppl) ~ label + (1|speaker) + (1|conversation)\n")
cat(strrep("=", 60), "\n")
print(summary(m_mbert))
cat("\n\n")
cat(strrep("=", 60), "\n")
cat("XLM-R MIXED MODEL\n")
cat("Formula: log(xlmr_pppl) ~ label + (1|speaker) + (1|conversation)\n")
cat(strrep("=", 60), "\n")
print(summary(m_xlmr))
sink()
cat("Saved: mixed_model_summary.txt\n")


# ── Extract and report key statistics

extract_fixed <- function(model, model_name) {
  coefs <- summary(model)$coefficients
  row   <- coefs["labelInvalid", ]
  vc    <- as.data.frame(VarCorr(model))

  list(
    model            = model_name,
    beta_invalid     = round(row["Estimate"],    4),
    std_error        = round(row["Std. Error"],  4),
    t_value          = round(row["t value"],     4),
    df_satterthwaite = round(row["df"],          1),
    p_value          = round(row["Pr(>|t|)"],    6),
    var_speaker      = round(vc$vcov[vc$grp == "speaker"],  4),
    var_conversation = round(vc$vcov[vc$grp == "conversation"], 4),
    var_residual     = round(vc$vcov[vc$grp == "Residual"], 4)
  )
}

mb_fe <- extract_fixed(m_mbert, "mBERT")
xl_fe <- extract_fixed(m_xlmr,  "XLM-RoBERTa")

cat("\n── Fixed effect: labelInvalid coefficient ──\n")
for (fe in list(mb_fe, xl_fe)) {
  sig <- ifelse(fe$p_value < .001, "***",
         ifelse(fe$p_value < .01,  "**",
         ifelse(fe$p_value < .05,  "*", "")))
  cat(sprintf("%-12s  beta=%.4f  SE=%.4f  t=%.3f  df=%.0f  p=%.6f  %s\n",
              fe$model, fe$beta_invalid, fe$std_error,
              fe$t_value, fe$df_satterthwaite, fe$p_value, sig))
}

cat("\n── Random effects variance components ──\n")
cat(sprintf("mBERT:   var(speaker)=%.4f  var(conversation)=%.4f  var(residual)=%.4f\n",
            mb_fe$var_speaker, mb_fe$var_conversation, mb_fe$var_residual))
cat(sprintf("XLM-R:   var(speaker)=%.4f  var(conversation)=%.4f  var(residual)=%.4f\n",
            xl_fe$var_speaker, xl_fe$var_conversation, xl_fe$var_residual))

cat("\n── Interpretation ──\n")
cat("  beta_invalid > 0 means invalid sentences have HIGHER log(PPPL)\n")
cat("  i.e. models find invalid sentences less natural -- supports H1\n")
cat("  Speaker and conversation variance shows how much PPPL varies\n")
cat("  across speakers and conversations independently of EC validity.\n")

# Save summary CSV
summary_df <- bind_rows(as_tibble(mb_fe), as_tibble(xl_fe))
write_csv(summary_df, file.path(OUTPUT_DIR, "mixed_results_summary.csv"))
cat("\nSaved: mixed_results_summary.csv\n")

cat("\n", strrep("=", 60), "\n")
cat("DONE. Outputs in:", OUTPUT_DIR, "/\n")
cat("  normality_check.png/.pdf\n")
cat("  boxplot_log.png/.pdf\n")
cat("  mixed_model_summary.txt\n")
cat("  mixed_results_summary.csv\n")
