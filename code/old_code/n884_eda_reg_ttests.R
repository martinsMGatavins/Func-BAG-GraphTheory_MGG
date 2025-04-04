invisible(lapply(c("stringi","stringr",
                   "tidyr","tidyverse","reshape2",
                   "nptest","ggplot2", "boot",
                   'colorspace','ggsci','visreg','QuantPsyc',
                   'corrplot'), require, character.only=TRUE))

ROOT <- "~/Desktop/BIDS_WUSTL_2022"
brain_age <- read.csv(paste0(ROOT,"/brainage_allmodels.csv")) %>%
  dplyr::select(!contains("_raw"))
deltas <- brain_age %>% mutate(across(bc_10Fold_brainage:reho_10Fold_brainage,~.-real_age))
load(file=paste0(ROOT,"/second_set/total_psych.RData"))
deltas <- merge(deltas,total_merged[,c("sub","MeanDisProc")],by="sub")
psych_withage <- total_merged[,c("sub","age_parent","gender_parent","internal_parent","internal_child",
                               "external_parent","external_child","total_parent","total_child","MeanDisProc")] %>%
  rename(age = age_parent,sex = gender_parent)
psych_byscore <- read.csv(file=paste0(ROOT,"/psych_byscore.csv")) %>%
  merge(deltas,by="sub") %>%
  dplyr::select(!c("age_ysra"))
psych_byscore <- psych_byscore %>% mutate(across(bc_10Fold_brainage:reho_10Fold_brainage,~resid(lm(.~age,data=psych_byscore))))
 
psych_clean <- psych_withage %>% dplyr::select(!c("age"))
psych_age <- merge(psych_clean,deltas,by="sub")
#psych_age <- merge(psych_clean,brain_age,by="sub")
#psych_delta <- merge(psych_clean,deltas,by="sub")

###
# BRAIN AGE DELTAS ~ CHRONOLOGICAL AGE (age bias)
#####
del_long <- melt(deltas,id.vars=c("sub","real_age"))
theme_set(theme_classic())
g0 <- ggplot(del_long,aes(x=real_age,y=value,color=variable))
g0 + geom_smooth(method="lm",se=F) + geom_hline(yintercept=0) +
  scale_color_brewer(palette="Paired")

lm1 <- lm(bc_10Fold_brainage ~ real_age,data=deltas)
summary(lm1) # corr = -0.69, p < 2 * 10^-16

lm2 <- lm(interfc_10Fold_brainage ~ real_age,data=deltas)
summary(lm2) # corr = -0.52, p < 2 * 10^-16

lm3 <- lm(part_10Fold_brainage ~ real_age,data=deltas)
summary(lm3) # corr = -0.55, p < 2 * 10^-16

lm4 <- lm(intrafc_10Fold_brainage ~ real_age,data=deltas)
summary(lm4) # corr = -0.45, p < 2 * 10^-16

lm5 <- lm(clust_10Fold_brainage ~ real_age,data=deltas)
summary(lm5) # corr = -0.48, p < 2 * 10^-16

lm6 <- lm(alff_10Fold_brainage ~ real_age,data=deltas)
summary(lm6) # corr = -0.45, p < 2 * 10^-16

lm7 <- lm(eff_10Fold_brainage ~ real_age,data=deltas)
summary(lm7) # corr = -0.485, p < 2 * 10^-16

lm8 <- lm(str_10Fold_brainage ~ real_age,data=deltas)
summary(lm8) # corr = -0.43, p < 2 * 10^-16

lm9 <- lm(reho_10Fold_brainage ~ real_age,data=deltas)
summary(lm9) # corr = -0.48, p < 2 * 10^-16
#####

###
# PSYCHOPATHOLOGY SCORE distributions (histograms)
#####
psych_melted <- melt(psych_withage,id.vars=c("sub","sex","age"))
internal_long <- filter(psych_melted,grepl("internal",variable))
g1 <- ggplot(internal_long,aes(value,fill=variable))
g1 + geom_histogram(position="dodge",binwidth = 1) +
  scale_fill_uchicago() +
  theme(legend.position = "top")

external_long <- filter(psych_melted,grepl("external",variable))
g2 <- ggplot(external_long,aes(value,fill=variable))
g2 + geom_histogram(position="dodge",binwidth = 1) +
  scale_fill_uchicago() +
  theme(legend.position = "top")

total_long <- filter(psych_melted,grepl("total",variable))
g3 <- ggplot(external_long,aes(value,fill=variable))
g3 + geom_histogram(position="dodge",binwidth = 1) +
  scale_fill_uchicago() +
  theme(legend.position = "top")

# by age groups (8-10, 11-14, 15-17, 19-22)
# for each questionnaire - 6 histogram plots (OR a boxplot with 4 x 6 bars)
psychlong_withage <- psych_melted %>% mutate(age_group = case_when(
  age < 13 ~ "8-12",
  age < 18 & age >= 13 ~ "13-17",
  TRUE ~ "18-22"
))

internal_byage <- filter(psychlong_withage,grepl("internal",variable))
internal_parent <- internal_byage %>% filter(grepl("parent",variable))
internal_child <- internal_byage %>% filter(grepl("child",variable))
internal_total <- merge(internal_parent %>% dplyr::select(sub,value,age_group) %>%
                          rename(internal_parent=value),
                        internal_child %>% dplyr::select(sub,value) %>%
                          rename(internal_child=value),on="sub")
internal_total$age_group <- factor(internal_total$age_group,
                                   levels=c("8-12","13-17","18-22"))

g4parent <- ggplot(internal_parent,aes(value,fill=age_group))
g4child <- ggplot(internal_child,aes(value,fill=age_group))

g4parent + geom_histogram(position="dodge",binwidth = 1) +
  scale_fill_uchicago() +
  theme(legend.position = "top") +
  facet_grid(factor(age_group,levels=c("8-12","13-17","18-22"))~.) +
  xlab("internal_parent")

g4child + geom_histogram(position="dodge",binwidth = 1) +
  scale_fill_uchicago() +
  theme(legend.position = "top") +
  facet_grid(factor(age_group,levels=c("8-12","13-17","18-22"))~.) +
  xlab("internal_child")

external_byage <- filter(psychlong_withage,grepl("external",variable))
external_parent <- filter(external_byage,grepl("parent",variable))
external_child <- filter(external_byage,grepl("child",variable)) 
external_total <- merge(external_parent %>% dplyr::select(sub,value,age_group) %>%
                          rename(external_parent=value),
                        external_child %>% dplyr::select(sub,value) %>%
                          rename(external_child=value),on="sub")
external_total$age_group <- factor(external_total$age_group,
                                   levels=c("8-12","13-17","18-22"))

g5parent <- ggplot(external_parent,aes(value,fill=age_group))
g5child <- ggplot(external_child,aes(value,fill=age_group))

g5parent + geom_histogram(position="dodge",binwidth = 1) +
  scale_fill_uchicago() +
  theme(legend.position = "top") +
  facet_grid(factor(age_group,levels=c("8-12","13-17","18-22"))~.) +
  xlab("external_parent")

g5child + geom_histogram(position="dodge",binwidth = 1) +
  scale_fill_uchicago() +
  theme(legend.position = "top") +
  facet_grid(factor(age_group,levels=c("8-12","13-17","18-22"))~.) +
  xlab("external_child")

total_byage <- filter(psychlong_withage,grepl("total",variable))
total_parent <- filter(total_byage,grepl("parent",variable))
total_child <- filter(total_byage,grepl("child",variable)) 
total_total <- merge(total_parent %>% dplyr::select(sub,value,age_group) %>%
                          rename(total_parent=value),
                     total_child %>% dplyr::select(sub,value) %>%
                          rename(total_child=value),on="sub")
total_total$age_group <- factor(total_total$age_group,
                                   levels=c("8-12","13-17","18-22"))

g6parent <- ggplot(total_parent,aes(value,fill=age_group))
g6child <- ggplot(total_child,aes(value,fill=age_group))

g6parent + geom_histogram(position="dodge",binwidth = 1) +
  scale_fill_uchicago() +
  theme(legend.position = "top") +
  facet_grid(factor(age_group,levels=c("8-12","13-17","18-22"))~.) +
  xlab("total_parent")

g6child + geom_histogram(position="dodge",binwidth = 1) +
  scale_fill_uchicago() +
  theme(legend.position = "top") +
  facet_grid(factor(age_group,levels=c("8-12","13-17","18-22"))~.) +
  xlab("total_child")

#####

# clean psych_age table:
colnames(psych_age) <- colnames(psych_age) %>% gsub("_10Fold_brainage","",.)
colnames(psych_byscore) <- colnames(psych_byscore) %>% gsub("_10Fold_brainage","",.)
psych_byscore <- rename(psych_byscore,fd = MeanDisProc)
psych_age <- rename(psych_age, fd = MeanDisProc, age = real_age)

###
# DELTA ~ AGE + SEX + SCORE + FD (compiled scores)
#####
agelm1_int <- lm(bc ~ age + sex + internal_child + fd, psych_age) # r < 0.001, p > .75
agelm1_ext <- lm(bc ~ age + sex + external_child + fd, psych_age) # r = 0.01, p = .35
agelm1_tot <- lm(bc ~ age + sex + total_child + fd, psych_age) # r = -0.003, p > .5
summary(agelm1_int);summary(agelm1_ext);summary(agelm1_tot)
lm.beta(agelm1_int);lm.beta(agelm1_ext);lm.beta(agelm1_tot)

agelm2_int <- lm(interfc ~ age + sex + internal_child + fd, psych_age) # r = 0.01, p = .17
agelm2_ext <- lm(interfc ~ age + sex + external_child + fd, psych_age) # r = -0.001, p > .75
agelm2_tot <- lm(interfc ~ age + sex + total_child + fd, psych_age) # r = 0.002, p > .75
summary(agelm2_int);summary(agelm2_ext);summary(agelm2_tot)
lm.beta(agelm2_int);lm.beta(agelm2_ext);lm.beta(agelm2_tot)

agelm3_int <- lm(part ~ age + sex + internal_child + fd, psych_age) # r = 0.01, p = .16
agelm3_ext <- lm(part ~ age + sex + external_child + fd, psych_age) # r = 0.02, p = .16
agelm3_tot <- lm(part ~ age + sex + total_child + fd, psych_age) # r = 0.001, p > .75
summary(agelm3_int);summary(agelm3_ext);summary(agelm3_tot)
lm.beta(agelm3_int);lm.beta(agelm3_ext);lm.beta(agelm3_tot)

agelm4_int <- lm(intrafc ~ age + sex + internal_child + fd, psych_age) # r = 0.01, p = .32
agelm4_ext <- lm(intrafc ~ age + sex + external_child + fd, psych_age) # r = 0.01, p = .35
agelm4_tot <- lm(intrafc ~ age + sex + total_child + fd, psych_age) # r = -0.002, p > .75
summary(agelm4_int);summary(agelm4_ext);summary(agelm4_tot)
lm.beta(agelm4_int);lm.beta(agelm4_ext);lm.beta(agelm4_tot)

agelm5_int <- lm(clust ~ age + sex + internal_child + fd, psych_age) # r = -0.001, p > .75
agelm5_ext <- lm(clust ~ age + sex + external_child + fd, psych_age) # r = 0.02, p = .1 (marginal)
agelm5_tot <- lm(clust ~ age + sex + total_child + fd, psych_age) # r = -0.003, p > .5
summary(agelm5_int);summary(agelm5_ext);summary(agelm5_tot)
lm.beta(agelm5_int);lm.beta(agelm5_ext);lm.beta(agelm5_tot)

# agelm5_ext - sensitivity (splitting by questionnaire makes the correlation fall apart)
lm5_ext_cbcl <- lm(clust ~ age + gender + external_scale_sch + fd, psych_byscore) # p =.670
lm5_ext_ysra <- lm(clust ~ age + gender + ysra_extnl_scale + fd, psych_byscore) #p=.927
lm5_ext_asr <- lm(clust ~ age + gender + asr_syn_extnl + fd, psych_byscore) #p=.804

summary(lm5_ext_cbcl); summary(lm5_ext_ysra); summary(lm5_ext_asr)

agelm6_int <- lm(alff ~ age + sex + internal_child + fd, psych_age) # r = 0.02, p = .08 (marginal)
agelm6_ext <- lm(alff ~ age + sex + external_child + fd, psych_age) # r = 0.02, p = .16
agelm6_tot <- lm(alff ~ age + sex + total_child + fd, psych_age) # r = 0.003, p > .5
summary(agelm6_int);summary(agelm6_ext);summary(agelm6_tot)
lm.beta(agelm6_int);lm.beta(agelm6_ext);lm.beta(agelm6_tot)

# agelm6_int

agelm7_int <- lm(eff ~ age + sex + internal_child + fd, psych_age) # r = 0.018, p = .057 (marginal)
agelm7_ext <- lm(eff ~ age + sex + external_child + fd, psych_age) # r = 0.026, p = .029 (*)
agelm7_tot <- lm(eff ~ age + sex + total_child + fd, psych_age) # r = 0.01, p = .15
summary(agelm7_int);summary(agelm7_ext);summary(agelm7_tot)
lm.beta(agelm7_int);lm.beta(agelm7_ext);lm.beta(agelm7_tot)

# agelm7_ext (* - n.s after sensitivity analyses)
lm7_ext_cbcl <- lm(eff ~ age + gender + external_scale_sch + fd, psych_byscore) # p =.978
lm7_ext_ysra <- lm(eff ~ age + gender + ysra_extnl_scale + fd, psych_byscore) #p=.198
lm7_ext_asr <- lm(eff ~ age + gender + asr_syn_extnl + fd, psych_byscore) #p=.261
summary(lm7_ext_cbcl); summary(lm7_ext_ysra); summary(lm7_ext_asr) # all no longer significant


agelm8_int <- lm(str ~ age + sex + internal_child + fd, psych_age) # r = 0.018, p = .052 (marginal)
agelm8_ext <- lm(str ~ age + sex + external_child + fd, psych_age) # r = 0.023, p = .054 (marginal)
agelm8_tot <- lm(str ~ age + sex + total_child + fd, psych_age) # r = 0.13, p = .33
summary(agelm8_int);summary(agelm8_ext);summary(agelm8_tot)
lm.beta(agelm8_int);lm.beta(agelm8_ext);lm.beta(agelm8_tot)

agelm9_int <- lm(reho ~ age + sex + internal_child + fd, psych_age) # r = 0.035, p < .001 (***)
agelm9_ext <- lm(reho ~ age + sex + external_child + fd, psych_age) # r = 0.02, p = .17
agelm9_tot <- lm(reho ~ age + sex + total_child + fd, psych_age) # r = 0.01, p = .14
summary(agelm9_int);summary(agelm9_ext);summary(agelm9_tot)
lm.beta(agelm9_int);lm.beta(agelm9_ext);lm.beta(agelm9_tot)

# agelm9_int
lm9_ext_cbcl <- lm(reho ~ age + gender + internal_scale_sch + fd, psych_byscore) # p =<.001
lm9_ext_ysra <- lm(reho ~ age + gender + ysra_intnl_scale + fd, psych_byscore) #p=.001 (***)
lm9_ext_asr <- lm(reho ~ age + gender + asr_syn_intnl + fd, psych_byscore) #p=.403

summary(lm9_ext_cbcl); summary(lm9_ext_ysra); summary(lm9_ext_asr)

#####

###
# DELTA ~ AGE * SCORE + SEX + FD (compiled scores)
#####
agescorelm1_int <- lm(bc ~ age*internal_child + sex + fd, psych_age) # r = -0.002, p = .25
agescorelm1_ext <- lm(bc ~ age*external_child + sex + fd, psych_age) # r = -0.001, p = .7
agescorelm1_tot <- lm(bc ~ age*total_child + sex + fd, psych_age) # r ~ 0, p > .5
summary(agescorelm1_int);summary(agescorelm1_ext);summary(agescorelm1_tot)
lm.beta(agescorelm1_int);lm.beta(agescorelm1_ext);lm.beta(agescorelm1_tot)

agescorelm2_int <- lm(interfc ~ age*internal_child + sex + fd, psych_age) # r = -0.003, p = .22
agescorelm2_ext <- lm(interfc ~ age*external_child + sex + fd, psych_age) # r = -0.002, p = .47
agescorelm2_tot <- lm(interfc ~ age*total_child + sex + fd, psych_age) # r ~ 0, p > .5
summary(agescorelm2_int);summary(agescorelm2_ext);summary(agescorelm2_tot)
lm.beta(agescorelm2_int);lm.beta(agescorelm2_ext);lm.beta(agescorelm2_tot)

agescorelm3_int <- lm(part ~ age*internal_child + sex + fd, psych_age) # r = -0.006, p = .005 (**)
agescorelm3_ext <- lm(part ~ age*external_child + sex + fd, psych_age) # r = -0.002, p = .45
agescorelm3_tot <- lm(part ~ age*total_child + sex + fd, psych_age) # r ~ 0, p > .5
summary(agescorelm3_int);summary(agescorelm3_ext);summary(agescorelm3_tot)
lm.beta(agescorelm3_int);lm.beta(agescorelm3_ext);lm.beta(agescorelm3_tot)

agescorelm4_int <- lm(intrafc ~ age*internal_child + sex + fd, psych_age) # r = -0.009, p < 0.001 (***)
agescorelm4_ext <- lm(intrafc ~ age*external_child + sex + fd, psych_age) # r = -0.004, p = .22
agescorelm4_tot <- lm(intrafc ~ age*total_child + sex + fd, psych_age) # r = -0.001, p = .35
summary(agescorelm4_int);summary(agescorelm4_ext);summary(agescorelm4_tot)
lm.beta(agescorelm4_int);lm.beta(agescorelm4_ext);lm.beta(agescorelm4_tot)

agescorelm5_int <- lm(clust ~ age*internal_child + sex + fd, psych_age) # r = -0.008, p < .001 (***)
agescorelm5_ext <- lm(clust ~ age*external_child + sex + fd, psych_age) # r = -0.008, p = .008 (**)
agescorelm5_tot <- lm(clust ~ age*total_child + sex + fd, psych_age) # r = -0.003, p = .018 (*)
summary(agescorelm5_int);summary(agescorelm5_ext);summary(agescorelm5_tot)
lm.beta(agescorelm5_int);lm.beta(agescorelm5_ext);lm.beta(agescorelm5_tot)

agescorelm6_int <- lm(alff ~ age*internal_child + sex + fd, psych_age) # r = -0.006, p = .01 (*)
agescorelm6_ext <- lm(alff ~ age*external_child + sex + fd, psych_age) # r = -0.004, p = .12
agescorelm6_tot <- lm(alff ~ age*total_child + sex + fd, psych_age) # r ~ 0, p > .5
summary(agescorelm6_int);summary(agescorelm6_ext);summary(agescorelm6_tot)
lm.beta(agescorelm6_int);lm.beta(agescorelm6_ext);lm.beta(agescorelm6_tot)

agescorelm7_int <- lm(eff ~ age*internal_child + sex + fd, psych_age) # r = -0.007, p = .003 (**)
agescorelm7_ext <- lm(eff ~ age*external_child + sex + fd, psych_age) # r = -0.004, p = .23
agescorelm7_tot <- lm(eff ~ age*total_child + sex + fd, psych_age) # r = -0.001, p = .35
summary(agescorelm7_int);summary(agescorelm7_ext);summary(agescorelm7_tot)
lm.beta(agescorelm7_int);lm.beta(agescorelm7_ext);lm.beta(agescorelm7_tot)

agescorelm8_int <- lm(str ~ age*internal_child + sex + fd, psych_age) # r = -0.006, p = .004 (**)
agescorelm8_ext <- lm(str ~ age*external_child + sex + fd, psych_age) # r = -0.007, p = .01 (*)
agescorelm8_tot <- lm(str ~ age*total_child + sex + fd, psych_age) # r = -0.003, p = 0.077 (marginal)
summary(agescorelm8_int);summary(agescorelm8_ext);summary(agescorelm8_tot)
lm.beta(agescorelm8_int);lm.beta(agescorelm8_ext);lm.beta(agescorelm8_tot)

agescorelm9_int <- lm(reho ~ age*internal_child + sex + fd, psych_age) # r = -0.009, p < .001 (***)
agescorelm9_ext <- lm(reho ~ age*external_child + sex + fd, psych_age) # r = -0.003, p = .24
agescorelm9_tot <- lm(reho ~ age*total_child + sex + fd, psych_age) # r ~ 0, p > .75
summary(agescorelm9_int);summary(agescorelm9_ext);summary(agescorelm9_tot)
lm.beta(agescorelm9_int);lm.beta(agescorelm9_ext);lm.beta(agescorelm9_tot)

#####

###
# PLOTTING SIGNIFICANT & MARGINALLY SIGNIFICANT CORRELATIONS
#####
# Significant
visreg(agelm7_ext,"external_child")

visreg(agelm9_int,"internal_child")

visreg(agescorelm3_int,"internal_child",by="age",band=F,overlay=T)

visreg(agescorelm4_int,"internal_child",by="age",band=F,overlay=T)

visreg(agescorelm5_int,"internal_child",by="age",band=F,overlay=T)
visreg(agescorelm5_ext,"external_child",by="age",band=F,overlay=T)
visreg(agescorelm5_tot,"total_child",by="age",band=F,overlay=T)

visreg(agescorelm6_int,"internal_child",by="age",band=F,overlay=T)

visreg(agescorelm7_int,"internal_child",by="age",band=F,overlay=T)

visreg(agescorelm8_int,"internal_child",by="age",band=F,overlay=T)
visreg(agescorelm8_ext,"external_child",by="age",band=F,overlay=T)

visreg(agescorelm9_int,"internal_child",by="age",band=F,overlay=T)

# Marginal
visreg(agelm5_ext,"external_child")
visreg(agelm6_int,"internal_child")
visreg(agelm7_int,"internal_child")
visreg(agelm8_int,"internal_child")
visreg(agelm8_ext,"external_child")

visreg(agescorelm8_tot,"total_child",by="age",band=F,overlay=T) # marginal

#####

###
# SENSITIVITY ANALYSES
#####
# Testing only significant interactions

# Suggestions on how to do the sensitivity analyses:
### remove adults? 
### by questionnaire?
### sampling without replacement?

#####


