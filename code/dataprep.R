library(tidyr)
library(dplyr)
library(stringr)
library(stringi)

# Cleaning cognitive data
cog_data <- read.csv(file = "data/hcp_d/HCD_NIH-Toolbox-Scores_2022-01-28.csv")
cog_data <- cog_data %>% filter(grepl("Total",Inst) & !grepl("Crystallized|Fluid",Inst))
cog_data <- cog_data[c("PIN","Age.Corrected.Standard.Score")]
cog_data <- cog_data %>% rename(sub = PIN, cog_score = Age.Corrected.Standard.Score)

# Extracting subset of data
large_data <- read.csv(file = "data/hcp_d/freeze_1688x2339_20230524.csv")
large_data <- large_data[large_data$Incl2BL_20220816 == "Include", ] # 1085
qc_data <- large_data[,c("sub","SUB","gender","Incl2BL_20220816","age","MeanDisOrig","NumVolProc")]
psych_data <- large_data[,c("sub","pfact_score","pathology_scale")]

#df_quality <- read.csv(file = "data/hcp_d/qa_1688x15_20240627.csv")
#df_psych <- read.csv(file = "data/hcp_d/psych_1414x8_20220718.csv")
#df_cog <- read.csv(file = "data/hcp_d/HCD_NIH-Toolbox-Scores_2022-01-28.csv")

# Scan quality filtering
# large_data <- large_data[complete.cases(large_data),]

iqr = IQR(qc_data$MeanDisOrig,na.rm=TRUE)
median = median(qc_data$MeanDisOrig,na.rm=TRUE)
fd_cutoff <- (median + iqr) * 1.5
strict_fd_cutoff <- median + iqr

qc_data$cut <- qc_data$MeanDisOrig >= fd_cutoff
qc_data$strict_cut <- qc_data$MeanDisOrig >= strict_fd_cutoff

# Check with table
table(qc_data$cut) # 953
table(qc_data$strict_cut) # 873

full_data <- left_join(qc_data,psych_data,by="sub")
full_data <- left_join(full_data,cog_data,by="sub")
full_data <- full_data[!is.na(full_data$cog_score) & !is.na(full_data$pfact_score),]

# Final sample size: N = 881, N_strict = 809
table(full_data$cut) # 881
table(full_data$strict_cut) # 809

write.csv(file="data/phen_scanqa_n998_090524.csv",full_data)
