library(dplyr)
library(ggplot2)
library(ggsci)
library(ggpubr)
library(rstatix)
library(cowplot)

theme_nb <- function(
    base_size = 9.5,
    base_family = ""
) {
  theme_classic(base_size = base_size, base_family = base_family) %+replace%
    theme(
      # Overall text
      text = element_text(family = base_family),
      
      # Panel border/axes: classic keeps panel background clean; we add axis lines explicitly
      axis.line = element_line(linewidth = 0.9, colour = "black"),
      
      # Tick marks
      axis.ticks = element_line(linewidth = 0.8, colour = "black"),
      axis.ticks.length = unit(2.0, "mm"),
      
      # Axis text (tick labels)
      axis.text = element_text(
        size = base_size,
        colour = "black",
        family = base_family
      ),
      
      # Axis titles (label size)
      axis.title = element_text(
        size = base_size * 1.3,
        face = "plain",
        colour = "black",
        family = base_family
      ),
      
      # Remove extra grid lines for a clean “publishing” look
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      
      # Margins (often helps multi-panel figures)
      plot.margin = margin(6, 10, 6, 10)
    )
}

# Comparing models-------------------------------------------------------
meta <- read.csv("C:/Users/test_set_log.csv") #Test set log file as shown in https://doi.org/10.5281/zenodo.21297859
colnames(meta)[1] <- "sample_name"

files <- list.files("C:/Users/pred",   #Import prediction table of five replicates
                    pattern = "\\.csv$", full.names = TRUE)
names(files) <- basename(files)
df_list <- lapply(files, read.csv)

for (i in names(df_list)){
  df_list[[i]]$rep <- sub(".*rep([0-9]+)\\.csv$", "\\1", i)
  df_list[[i]] <- merge(meta, df_list[[i]], by = "sample_name")
}
data <- do.call(rbind, df_list)

# Calculate ape
ape <- function(y_true, y_pred) {
  abs((y_pred - y_true )) / y_true * 100
}

data$ape <- ape(data$individual_count_estimation, data$point_pred)

ggplot(data, aes(samp, ape))+
  geom_boxplot(aes(samp, ape, colour = rep),width = 0.15,
               position = position_dodge(width = 0.9))


dat_summ <- data %>%
  group_by(rep)%>%
  summarise(sd_sq = sum((individual_count_estimation - point_pred)^2))%>%
  ungroup()



# Final model test plot----------------------------------------------
setwd("C:/Users/")
meta <- read.csv("test_set_log.csv")
ref <- read.csv("./v10_rep2.csv")  #The best rep
colnames(meta)[1] <- "sample_name"

ref <- merge(meta, ref, by = "sample_name")
lm(individual_count_estimation ~ point_pred,data = ref)

ggplot(ref, aes(individual_count_estimation, point_pred))+
  geom_pointrange(aes(ymin = range_low, ymax = range_high), 
                  position=position_jitter(width= 1), alpha = 0.6) +
  geom_abline(slope = 1, intercept = 0, color = "red",linewidth = 0.8 ,alpha = 0.4)+
  xlab("Individual Number")+
  ylab("Predicted Number")+
  theme_nature()


resid <- data.frame("resid" = ref$point_pred - ref$individual_count_estimation, 
                    "resid_low" = ref$range_low - ref$individual_count_estimation,
                    "resid_high" = ref$range_high - ref$individual_count_estimation,
                    "true_num" = ref$individual_count_estimation,
                    "predicted_num" = ref$point_pred)
resid$ape <- abs(resid$resid)/resid$true_num
resid$pred_range <- resid$resid_high - resid$resid_low

which(resid$resid_low > 0 | resid$resid_high <0)


ggplot(resid, aes(true_num, resid / true_num))+
  geom_pointrange(aes(ymin = resid_low/true_num , ymax = resid_high/true_num), 
                  position=position_jitter(width= 1), alpha = 0.6) +
  geom_abline(slope = 0, intercept = 0, color = "red")

  



ref$resid <- ref$point_pred - ref$individual_count_estimation
ref$fm_ratio <- ref$female_count / ref$male_count


colnames(ref)[9] <- "background_noise"
bias_model <- lm(resid ~ individual_count_estimation + detection_clip_number + clips_overlapping_calls_proportion + fm_ratio +
     average_peak_volumn + background_noise,
   data = ref)
precision_model <- lm(abs(resid) ~ individual_count_estimation + detection_clip_number + clips_overlapping_calls_proportion + fm_ratio +
                        average_peak_volumn + background_noise,
                      data = ref)
bias_model_coe <-  as.data.frame(summary(bias_model)$coefficients)
bias_model_coe$factor <- rownames(bias_model_coe)
precision_model_coe <- as.data.frame(summary(precision_model)$coefficients)
precision_model_coe$factor <- rownames(precision_model_coe)
models <- rbind(bias_model_coe, precision_model_coe)
models$model <- c(rep("bias",7), rep("precision",7))
models <- models[-c(1,8),]

colnames(models)

models[which(models$`Pr(>|t|)`> 0.05), 4] <- "n.s."
models[which(models$`Pr(>|t|)`<= 0.05 & models$`Pr(>|t|)` > 0.01), 4] <- "*"
models[which(models$`Pr(>|t|)`<= 0.01 & models$`Pr(>|t|)` > 0.001), 4] <- "**"
models[which(models$`Pr(>|t|)`<= 0.0001 ), 4] <- "***"
models$`Pr(>|t|)` <- factor(models$`Pr(>|t|)`, levels = c("n.s.", "*", "**", "***"))
models$Sign <- sign(models$Estimate)
models$Sign[which(models$Sign == "-1")] <- "Negative"
models$Sign[which(models$Sign == "1")] <- "Possitive"
models$Sign <- factor(models$Sign, levels = c("Negative", "Possitive"))
models <- models %>%
  mutate(factor = recode(factor, individual_count_estimation = "individual_count", 
                         clips_overlapping_calls_proportion = "overlapping call ratio",
                         fm_ratio = "female/male ratio",
                         average_peak_volumn = "average peak volume"))  
models$model[which(models$model == "bias")] <- "error"
models$model[which(models$model == "precision")] <- "abs. error"
models$model <- factor(models$model, levels = c("error", "abs. error"))


# Linear regression results plot
ggplot(models, aes(model, factor))+
  geom_point(aes(colour = Sign , size = abs(Estimate), alpha =`Pr(>|t|)`))+
  scale_y_discrete(drop = FALSE)+
  scale_size_continuous(range = c(2, 12))+
  labs(size = "Estimate", colour = "Sign", alpha ="Significance")+
  xlab("Prediction Error")+
  ylab("Factor")+
  scale_colour_npg()+
  theme_nb()+
  theme(
    panel.grid.major = element_line(linetype = "dashed", colour = "lightgrey"),
    panel.grid.minor = element_line(linetype = "dashed",colour = "lightgrey")
  )
