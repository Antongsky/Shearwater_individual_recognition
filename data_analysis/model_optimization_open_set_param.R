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

setwd("C:/Users/pred")

# Comparing models recog_threshold ---------------------------------------------

meta <- read.csv("C:/Users/test_set_log.csv")
colnames(meta)[1] <- "sample_name"

files <- list.files("C:/Users/test_rec_threshold", 
                    pattern = "\\.csv$", full.names = TRUE)
names(files) <- basename(files)
df_list <- lapply(files, read.csv)

for (i in names(df_list)){
  
  df_list[[i]]$test_samp <- sub("^testSampCtl([^_]+)_.*$", "\\1", i)               
  df_list[[i]]$rep <- sub("^.*_([^_]+)\\.csv$", "\\1", i)
  df_list[[i]] <- merge(meta, df_list[[i]], by = "sample_name")
}
data <- do.call(rbind, df_list)

ape <- function(y_true, y_pred) {
  abs((y_pred - y_true )) / y_true * 100
}

data$ape <- ape(data$individual_count_estimation, data$point_pred)

dat_summ <- data %>%
  group_by(test_samp, rep)%>%
  summarise(sd_sq = sum((individual_count_estimation - point_pred)^2))%>%
  ungroup()

kruskal_test(sd_sq ~ test_samp, data =dat_summ)

p1 <- ggplot(dat_summ, aes(test_samp, sd_sq))+
  geom_boxplot(linewidth = 0.1)+
  geom_point()+
  labs(x = "Test Set BirdNet Recognition Threshold", y = "SSR")+
  stat_compare_means(data = dat_summ,aes(test_samp, sd_sq),
                     method = "kruskal.test",
                     label = "p.format",
                     label.y = 4500,size = 4)+
  scale_color_npg()+
  theme_nb()






# Comparing models sampling ctl ---------------------------------------------

meta <- read.csv("C:/Users/test_set_log.csv")
colnames(meta)[1] <- "sample_name"

files <- list.files("C:/Users/sampling_ctl_v3", 
                    pattern = "\\.csv$", full.names = TRUE)
names(files) <- basename(files)
df_list <- lapply(files, read.csv)

for (i in names(df_list)){
  
  df_list[[i]]$samp <- sub("^v([^_]+)_.*$", "\\1", i)               
  df_list[[i]]$rep <- sub("^.*_([^_]+)\\.csv$", "\\1", i)
  df_list[[i]] <- merge(meta, df_list[[i]], by = "sample_name")
}
data <- do.call(rbind, df_list)

data <- data %>%
  mutate(samp = recode(samp,
                       `1` = "12-48",
                       `4` = "12-max",
                       `5` = "20-48",
                       `6` = "12-36",
                       `7` = "20-24",
                       `8` = "20-max",
                       `9` = "12-15",
                       `10` = "12-20"))


ape <- function(y_true, y_pred) {
  abs((y_pred - y_true )) / y_true * 100
}

data$ape <- ape(data$individual_count_estimation, data$point_pred)

ggplot(data, aes(samp, ape))+
  geom_violin(aes(fill = rep), alpha = 0.3)+
  geom_boxplot(aes(samp, ape, colour = rep),width = 0.15,
               position = position_dodge(width = 0.9))
  

dat_summ <- data %>%
  group_by(samp, rep)%>%
  summarise(sd_sq = sum((individual_count_estimation - point_pred)^2))%>%
  ungroup()

kruskal_test(sd_sq ~ samp, data =dat_summ)

dunn_res <- dat_summ %>%
  dunn_test(sd_sq ~ samp, p.adjust.method = "BH")


p2 <- ggplot(dat_summ, aes(samp, sd_sq))+
  geom_boxplot(linewidth = 0.1)+
  geom_point()+
  labs(x = "Sample per Class", y = "SSR")+
  stat_compare_means(data = dat_summ,aes(samp, sd_sq),
                     method = "kruskal.test",
                     label = "p.format",
                     label.y = 3900,size = 4)+
  scale_color_npg()+
  theme_nb()



# Comparing models PCA --------------------------------------------------------
meta <- read.csv("C:/Users/26739/Desktop/新建文件夹 (16)/benchmark_open_set/pred/test_set_log.csv")
colnames(meta)[1] <- "sample_name"

files <- list.files("C:/Users/26739/Desktop/新建文件夹 (16)/benchmark_open_set/pred/PCA", 
                    pattern = "\\.csv$", full.names = TRUE)
names(files) <- basename(files)
df_list <- lapply(files, read.csv)

for (i in names(df_list)){
  
  df_list[[i]]$PCA_dim <- sub("^PCA([^_]+)_.*$", "\\1", i)               
  df_list[[i]]$rep <- sub("^.*_([^_]+)\\.csv$", "\\1", i)
  df_list[[i]] <- merge(meta, df_list[[i]], by = "sample_name")
}
data <- do.call(rbind, df_list)

ape <- function(y_true, y_pred) {
  abs((y_pred - y_true )) / y_true * 100
}

data$ape <- ape(data$individual_count_estimation, data$point_pred)


dat_summ <- data %>%
  group_by(PCA_dim, rep)%>%
  summarise(sd_sq = sum((individual_count_estimation - point_pred)^2))%>%
  ungroup()

kruskal_test(sd_sq ~ PCA_dim, data =dat_summ)

dunn_res <- dat_summ %>%
  dunn_test(sd_sq ~ PCA_dim, p.adjust.method = "BH")%>%
  filter(p.adj.signif != "ns")

dunn_res$y.position <- c(5600,5800) 

p3<- ggplot(dat_summ, aes(PCA_dim, sd_sq))+
  geom_boxplot(linewidth = 0.1)+
  geom_point()+
  stat_pvalue_manual(
    dunn_res,
    label = "p.adj.signif",
    tip.length = 0.01
  ) +
  labs(x = "PCA components", y = "SSR")+
  stat_compare_means(data = dat_summ,aes(PCA_dim, sd_sq),
                     method = "kruskal.test",
                     label = "p.format",
                     label.y = 6000,size = 4)+
  scale_color_npg()+
  theme_nb()



# Final plot---------------------------------------------------------------
plot_grid(p1, p2, p3, labels = "AUTO", ncol = 3)


# Final model test plot----------------------------------------------
meta <- read.csv("test_set_log.csv")
ref <- read.csv("pred_v3.csv")
colnames(meta)[1] <- "sample_name"

ref <- merge(meta, ref, by = "sample_name")

ggplot(ref, aes(individual_count_estimation, point_pred))+
  geom_pointrange(aes(ymin = range_low, ymax = range_high), 
                  position=position_jitter(width= 1), alpha = 0.6) +
  geom_abline(slope = 1, intercept = 0, color = "red")


resid <- data.frame("resid" = ref$point_pred - ref$individual_count_estimation, 
                    "resid_low" = ref$range_low - ref$individual_count_estimation,
                    "resid_high" = ref$range_high - ref$individual_count_estimation,
                    "true_num" = ref$individual_count_estimation)
ggplot(resid, aes(true_num, resid / true_num))+
  geom_pointrange(aes(ymin = resid_low/true_num , ymax = resid_high/true_num), 
                  position=position_jitter(width= 1), alpha = 0.6) +
  geom_abline(slope = 0, intercept = 0, color = "red")



