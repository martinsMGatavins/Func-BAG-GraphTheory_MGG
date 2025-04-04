library(stringi)
library(stringr)
library(tidyr)
library(dplyr)
library(ggplot2)
library(tidyverse)
library(reshape2)
library(nptest)
library(boot)
library(colorspace)
library(Hmisc)
library(ggsci)
library(visreg)
library(grid)
# library(ggseg)
# library(ggseg3d)
# library(ggsegGordon)
library(ggpubr)
# library(freesurfer)

ROOT <- "Desktop/BIDS_WUSTL_2022/manuscript/"
NONWEIGHT_ROOT <- paste0(ROOT,"data/prediction/outputs")
WEIGHT_ROOT <- paste0(NONWEIGHT_ROOT,"_weighted")
outcome_vars <- list.files(NONWEIGHT_ROOT)
cogcorr_vars <- outcome_vars[grepl("_corr",outcome_vars)]
cogucorr_vars <- outcome_vars[grepl("uncorr",outcome_vars)]
psych_vars <- outcome_vars[grepl("child",outcome_vars)]

gordon333parcelnames <- read.delim(paste0(ROOT,"gordon333/gordon333NodeNames.txt"),header=F)$V1
gordon333partition <- read.delim(paste0(ROOT,"gordon333/gordon333CommunityAffiliation.1D"),header=F)$V1
gordon333networknames <- read.delim(paste0(ROOT,"gordon333/gordon333CommunityNames.txt"),header=F)$V1

models <- c("alff","bcentr","clust","eff","intrafc","interfc","part","reho","str")

# DATA WRANGLING
#####
cogcorr_stats <- read.csv(paste0(NONWEIGHT_ROOT,"/",cogcorr_vars[1],"/",models[1],"/",cogcorr_vars[1],".csv")) %>% dplyr::select(sub)
cogucorr_stats <- read.csv(paste0(NONWEIGHT_ROOT,"/",cogucorr_vars[1],"/",models[1],"/",cogucorr_vars[1],".csv")) %>% dplyr::select(sub)
psych_stats <- read.csv(paste0(NONWEIGHT_ROOT,"/",psych_vars[1],"/",models[1],"/",psych_vars[1],".csv")) %>% dplyr::select(sub)
age_stats <- read.csv(paste0(NONWEIGHT_ROOT,"/age/",models[1],"/age.csv")) %>% dplyr::select(sub)

cogcorr_scores <- cogcorr_stats; cogucorr_scores <- cogucorr_stats; psych_scores <- psych_stats; age_scores <- age_stats
 
features <- read.csv(paste0(NONWEIGHT_ROOT,"/",cogcorr_vars[1],"/",models[1],"/",cogcorr_vars[1],"_globalFIs.csv")) %>% dplyr::select(c(features))
features$features <- gordon333parcelnames
features$features <- sub("L_","",features$features)
features$features <- sub("R_","",features$features)
cogcorr_features <- features; cogucorr_features <- features; psych_features <- features; age_features <- features

for (outcome in outcome_vars) {
  temp <- read.csv(paste0(NONWEIGHT_ROOT,"/",outcome,"/",models[1],"/",outcome,".csv"))
  temp[[outcome]] <- temp$real_score
  temp <- temp %>% dplyr::select(c('sub',outcome))
  
  if (outcome %in% cogcorr_vars) {
    cogcorr_scores <- merge(cogcorr_scores,temp,by="sub",all=T)
  } else if (outcome %in% cogucorr_vars) {
    cogucorr_scores <- merge(cogucorr_scores,temp,by="sub",all=T)
  } else if (outcome %in% psych_vars) {
    psych_scores <- merge(psych_scores,temp,by="sub",all=T)
  } else {
    age_scores <- merge(age_scores,temp,by="sub",all=T)
  }
  
  for (model in models) {
    temp <- read.csv(paste0(NONWEIGHT_ROOT,"/",outcome,"/",model,"/",outcome,".csv"))
    temp[[paste0(outcome,"_",model)]] <- temp$pred_score
    temp <- temp %>% dplyr::select(c('sub',paste0(outcome,"_",model)))
    
    ftemp <- read.csv(paste0(NONWEIGHT_ROOT,"/",outcome,"/",model,"/",outcome,"_globalFIs.csv"))
    #ftemp[[paste0(outcome,"_",model)]] <- ftemp$mdi
    #ftemp <- ftemp %>% dplyr::select(c('features',paste0(outcome,"_",model)))
    
    if (outcome %in% cogcorr_vars) {
      cogcorr_stats <- merge(cogcorr_stats,temp,by="sub",all=T)
      cogcorr_features[[paste0(outcome,"_",model)]] <- ftemp$mdi
    } else if (outcome %in% cogucorr_vars) {
      cogucorr_stats <- merge(cogucorr_stats,temp,by="sub",all=T)
      cogucorr_features[[paste0(outcome,"_",model)]] <- ftemp$mdi
    } else if (outcome %in% psych_vars) {
      psych_stats <- merge(psych_stats,temp,by="sub",all=T)
      psych_features[[paste0(outcome,"_",model)]] <- ftemp$mdi
    } else {
      age_stats <- merge(age_stats,temp,by="sub",all=T)
      age_features[[paste0(outcome,"_",model)]] <- ftemp$mdi
    }
    
    temp <- read.csv(paste0(WEIGHT_ROOT,"/",outcome,"/",model,"/",outcome,"_weighted.csv"))
    temp[[paste0(outcome,"_",model,"_weighted")]] <- temp$pred_score
    temp <- temp %>% dplyr::select(c('sub',paste0(outcome,"_",model,"_weighted")))
    
    ftemp <- read.csv(paste0(WEIGHT_ROOT,"/",outcome,"/",model,"/",outcome,"_weighted_globalFIs.csv"))
    #ftemp[[paste0(outcome,"_",model,"_weighted")]] <- ftemp$mdi
    #ftemp <- ftemp %>% dplyr::select(c('features',paste0(outcome,"_",model,"_weighted")))
    
    if (outcome %in% cogcorr_vars) {
      cogcorr_stats <- merge(cogcorr_stats,temp,by="sub",all=T)
      cogcorr_features[[paste0(outcome,"_",model,"_weighted")]] <- ftemp$mdi
    } else if (outcome %in% cogucorr_vars) {
      cogucorr_stats <- merge(cogucorr_stats,temp,by="sub",all=T)
      cogucorr_features[[paste0(outcome,"_",model,"_weighted")]] <- ftemp$mdi
    } else if (outcome %in% psych_vars) {
      psych_stats <- merge(psych_stats,temp,by="sub",all=T)
      psych_features[[paste0(outcome,"_",model,"_weighted")]] <- ftemp$mdi
    } else {
      age_stats <- merge(age_stats,temp,by="sub",all=T)
      age_features[[paste0(outcome,"_",model,"_weighted")]] <- ftemp$mdi
    }
  }
}

ccorr <- cogcorr_stats %>% dplyr::select(c(sub) | starts_with("crystallized_corr"))
cucorr <- cogucorr_stats %>% dplyr::select(c(sub) | starts_with("crystallized_uncorr"))
fcorr <- cogcorr_stats %>% dplyr::select(c(sub) | starts_with("fluid_corr"))
fucorr <- cogucorr_stats %>% dplyr::select(c(sub) | starts_with("fluid_uncorr"))
tcorr <- cogcorr_stats %>% dplyr::select(c(sub) | starts_with("tot_cognition_corr"))
tucorr <- cogucorr_stats %>% dplyr::select(c(sub) | starts_with("tot_cognition_uncorr"))

internal <- psych_stats %>% dplyr::select(c(sub) | starts_with("internal"))
external <- psych_stats %>% dplyr::select(c(sub) | starts_with("external"))
total <- psych_stats %>% dplyr::select(c(sub) | starts_with("total"))

age <- age_stats

modularity = c("clust","part")
centrality = c("bcentr","str")
connectivity = c("reho","intrafc","interfc")
#####

# BOOTSTRAPPING CONF INTERVALS FUNCTION
basic.boot <- function(x){
  mean.boot<-function(y,i){mean.boot<-mean(y[i])}
  b <- boot(x,mean.boot,R=10000)
  conf_ints <- boot.ci(b,conf=.95,type="basic")$basic[4:5]
  xmin <- conf_ints[1]
  xmax <- conf_ints[2]
  mean <- mean(b$t)
  data.frame(y=mean,ymin=xmin,ymax=xmax)
}

# BOOTSTRAPPING ZERO-INTERCEPT SLOPE ESTIMATION
slope.boot <- function(x){
  cor.boot<-function(y,i){cor.boot<-unname(coef(lm(real_score~pred_score,data=y[i,]))[2])}
  b<- boot(x,cor.boot,R=1000)
  conf_ints <- boot.ci(b,conf=.95,type="basic")$basic[4:5]
  xmin <- conf_ints[1]
  xmax <- conf_ints[2]
  k <- unname(coef(lm(real_score~pred_score,data=x))[2])
  data.frame(y=k,ymin=xmin,ymax=xmax)
}

# ABSOLUTE MAE + SLOPE PLOTTING FUNCTION
plot.maes.slopes <- function(outcome_df,title_text,predictor_string,score_df) {
  outcome_long <- melt(outcome_df,id.vars=c("sub"))
  outcome_long <- outcome_long %>% mutate(model = gsub("_.*","",variable),
                                          feature = sub("^[^_]*_", "",variable))
  outcome_long <- outcome_long %>% mutate(modality = case_when(
    feature %in% modularity ~ "graph modularity",
    feature %in% centrality ~ "graph centrality",
    feature == "eff" ~ "graph efficiency",
    feature %in% connectivity ~ "non-graph connectivity",
    feature == "alff" ~ "ALFF"
  ))
  
  colnames(score_df)[2] <- "real_score"
  outcome_long <- merge(outcome_long,score_df,by="sub")
  outcome_long$MAE <- abs(outcome_long$value - outcome_long$real_score)
  outcome_long <- dplyr::rename(outcome_long,pred_score=value)

  mae_boxes <- ggplot(data=outcome_long,aes(x=feature,y=MAE,fill=modality)) +
    stat_summary(fun.data=basic.boot,geom="crossbar",width=0.75) +
    theme(axis.text.x=element_text(angle=90)) +
    labs(y="Absolute MAE", x="Model by input feature",
         title = title_text,
         fill="Feature types") +
    scale_fill_brewer(palette="RdGy") +
    theme_classic() + 
    geom_jitter(stat="identity",size=0.5)

  slope_lines <- ggplot(data=outcome_long,aes(x=real_score,y=pred_score,color=feature)) +
    stat_smooth(method="lm",se=F) +
    labs(x = paste0("Actual ",predictor_string),
         y = paste0("Predicted ",predictor_string),
         color = "Model by input feature") +
    geom_abline(slope=1,linetype=2) +
    xlim(100,125) + ylim(100,125) +
    theme_classic()
  
  plot_list <- list("maeplot"=mae_boxes,
                    "slopeplot"=slope_lines)
  return(plot_list)
  #return(outcome_long)
}

# ABSOLUTE MAE COMPARISON PLOTS
plot.maecompboxes <- function(outcome_df,title_text,predictor_string,score_df) {
  outcome_long <- melt(outcome_df,id.vars=c("sub"))
  outcome_long <- outcome_long %>% mutate(weighted = grepl("_weighted",outcome_long$variable))
  outcome_long <- outcome_long %>% mutate(variable=gsub("_weighted","",variable))
  outcome_long <- outcome_long %>% mutate(model = gsub("_.*","",variable),
                                          feature = sub("^[^_]*_", "",variable))
  colnames(score_df)[2] <- "real_score"
  outcome_long <- merge(outcome_long,score_df,by="sub")
  outcome_long$MAE <- abs(outcome_long$value - outcome_long$real_score)
  outcome_long <- dplyr::rename(outcome_long,pred_score=value)
  
  mae_compboxes <- ggplot(outcome_long,aes(x=feature,y=MAE,fill=weighted)) +
    stat_summary(fun.data=basic.boot,geom="crossbar",width=0.75,position="dodge") +
    labs(y="Absolute MAE", x="Model by input feature",
         title = title_text,
         fill="Feature types") +
    scale_fill_brewer(palette="RdGy") +
    theme_classic()
  return(mae_compboxes)
}

# SLOPE BOXPLOT PLOT FUNCTION
plot.model.slopeboxes <- function(outcome_df,predictor_string,score_df) {
  outcome_long <- melt(outcome_df,id.vars=c("sub"))
  outcome_long <- outcome_long %>% mutate(weighted = grepl("_weighted",outcome_long$variable))
  outcome_long <- outcome_long %>% mutate(variable=gsub("_weighted","",variable))
  outcome_long <- outcome_long %>% mutate(model = gsub("_.*","",variable),
                                          feature = sub("^[^_]*_", "",variable))
  outcome_long <- outcome_long %>% mutate(modality = case_when(
    feature %in% modularity ~ "graph modularity",
    feature %in% centrality ~ "graph centrality",
    feature == "eff" ~ "graph efficiency",
    feature %in% connectivity ~ "non-graph connectivity",
    feature == "alff" ~ "ALFF"
  ))

  colnames(score_df)[2] <- "real_score"
  outcome_long <- merge(outcome_long,score_df,by="sub")
  outcome_long <- dplyr::rename(outcome_long,pred_score=value)
  
  slopes <- as.data.frame(matrix(data=NA,nrow=18,ncol=6))
  colnames(slopes) <- c("model","y","ymin","ymax","weighted","modality")
  for (m in 1:length(unique(outcome_long$feature))) {
    n = 2*m-1
    # non-weighted
    model_n <- unique(outcome_long$feature)[m]
    df <- outcome_long %>% filter(feature == model_n & weighted == FALSE)
    modality <- unique(df$modality)
    boot_slope_nw <- slope.boot(df)
    slopes[n,] <- cbind(model_n,boot_slope_nw,"nonweighted",modality)
    

    # weighted
    n = n + 1
    model_n <- unique(outcome_long$feature)[m]
    df <- outcome_long %>% filter(feature == model_n & weighted)
    boot_slope_w <- slope.boot(df)
    slopes[n,] <- cbind(model_n,boot_slope_w,"weighted",modality)
  }
  coefficient_boxes <- ggplot(data=slopes,aes(x=model,y=y,fill=factor(weighted))) +
    geom_crossbar(aes(ymin=ymin,ymax=ymax),width=0.85,position="dodge") +
    theme(axis.text.x=element_text(angle=90)) +
    labs(y="Predicted-real regression coefficient", x="Model by input feature",
         title = predictor_string,
         fill="Feature types") +
    scale_fill_brewer(palette="RdGy") +
    geom_hline(yintercept=0,linetype=2) +
    theme_classic()
  
  # plotting intercept boxes with CIs?
  # intercept_boxes <- ggplot(data=intercepts,aes(x=model,y=y,fill=factor(weighted))) +
  #   geom_crossbar(aes(ymin=ymin,ymax=ymax),width=0.85,position="dodge") +
  #   theme(axis.text.x=element_text(angle=90)) +
  #   labs(y="Predicted-real regression intercept", x="Model by input feature",
  #        title = predictor_string,
  #        fill="Feature types") +
  #   scale_fill_brewer(palette="RdGy") +
  #   theme_classic()
  return(coefficient_boxes)
}

# ABSOLUTE MAE COMPARISON PLOTS (UNFINISHED)
plot.weight.maecomp <- function(outcome_df,title_string,score_df) {
  
}

### MAE & SLOPE PLOTS (MAE and slope comparisons)
# (Figure 1. Error analysis & bias correction)
# (Supplemental figures to send in a different document: Slopes and their legibility)
# CRYSTALLIZED CORRECTED
#####
colnames(ccorr) <- gsub("crystallized_corr","crystallizedcorr",colnames(ccorr))
ccorr_uw <- dplyr::select(ccorr,!contains("_weighted"))
cryst_corr <- plot.maes.slopes(ccorr_uw,"Corrected crystallized cognition",
                               "crystallized cognition",cogcorr_scores[,c("sub","crystallized_corr")])
# cryst_corr$maeplot + geom_jitter(size=0.2) ### ADD THESE IN SUPPLEMENT
#cryst_corr$maeplot

cryst_corrslopes <- plot.model.slopeboxes(ccorr,"Corrected crystallized cognition",cogcorr_scores[,c("sub","crystallized_corr")])
#cryst_corrslopes
#####

# CRYSTALLIZED UNCORRECTED
#####
colnames(cucorr) <- gsub("crystallized_uncorr","crystallizeduncorr",colnames(cucorr))
cucorr_uw <- dplyr::select(cucorr,!contains("_weighted"))
cryst_uncorr <- plot.maes.slopes(cucorr_uw,"Uncorrected crystallized cognition",
                                 "crystallized cognition",cogucorr_scores[,c("sub","crystallized_uncorr")])
#cryst_uncorr$maeplot

cryst_uncorrslopes <- plot.model.slopeboxes(cucorr,"Uncorrected crystallized cognition",cogucorr_scores[,c("sub","crystallized_uncorr")])
#cryst_uncorrslopes
#####

# FLUID CORRECTED
#####
colnames(fcorr) <- gsub("fluid_corr","fluidcorr",colnames(fcorr))
fcorr_uw <- dplyr::select(fcorr,!contains("_weighted"))
fluid_corr <- plot.maes.slopes(fcorr_uw,"Corrected fluid cognition",
                               "fluid cognition",cogcorr_scores[,c("sub","fluid_corr")])
#fluid_corr$maeplot

fluid_corrslopes <- plot.model.slopeboxes(fcorr,"Corrected fluid cognition",cogcorr_scores[,c("sub","fluid_corr")])
#fluid_corrslopes
#####

# FLUID UNCORRECTED
#####
colnames(fucorr) <- gsub("fluid_uncorr","fluiduncorr",colnames(fucorr))
fucorr_uw <- dplyr::select(fucorr,!contains("_weighted"))
fluid_uncorr <- plot.maes.slopes(fucorr_uw,"Uncorrected fluid cognition",
                                 "fluid cognition",cogucorr_scores[,c("sub","fluid_uncorr")])
#fluid_uncorr$maeplot

fluid_uncorrslopes <- plot.model.slopeboxes(fucorr,"Uncorrected fluid cognition",cogucorr_scores[,c("sub","fluid_uncorr")])
#fluid_uncorrslopes
#####

# TOTAL CORRECTED
#####
colnames(tcorr) <- gsub("tot_cognition_corr","totcognitionorr",colnames(tcorr))
tcorr_uw <- dplyr::select(tcorr,!contains("_weighted"))
total_corr <- plot.maes.slopes(tcorr_uw,"Corrected total cognition",
                               "total cognition",cogcorr_scores[,c("sub","tot_cognition_corr")])
#total_corr$maeplot

total_corrslopes <- plot.model.slopeboxes(fcorr,"Corrected total cognition",cogcorr_scores[,c("sub","tot_cognition_corr")])
#total_corrslopes
#####

# TOTAL UNCORRECTED
#####
colnames(tucorr) <- gsub("tot_cognition_uncorr","totcognitionuncorr",colnames(tucorr))
tucorr_uw <- dplyr::select(tucorr,!contains("_weighted"))
total_uncorr <- plot.maes.slopes(tucorr_uw,"Uncorrected total cognition",
                                 "total cognition",cogucorr_scores[,c("sub","tot_cognition_uncorr")])
#total_uncorr$maeplot

total_uncorrslopes <- plot.model.slopeboxes(tucorr,"Uncorrected total cognition",cogucorr_scores[,c("sub","tot_cognition_uncorr")])
#total_uncorrslopes
#####

# INTERNALIZING
#####
colnames(internal) <- gsub("_child","",colnames(internal))
internal_uw <- dplyr::select(internal,!contains("_weighted"))
internal_plot <- plot.maes.slopes(internal_uw,"Internal syndrome score",
                                 "internal syndrome score",psych_scores[,c("sub","internal_child")])
#internal_plot$maeplot

internal_slopes <- plot.model.slopeboxes(internal,"Internal syndrome score",psych_scores[,c("sub","internal_child")])
#internal_slopes
#####

# EXTERNALIZING
#####
colnames(external) <- gsub("_child","",colnames(external))
external_uw <- dplyr::select(external,!contains("_weighted"))
external_plot <- plot.maes.slopes(external_uw,"External syndrome score",
                                  "External syndrome score",psych_scores[,c("sub","external_child")])
#external_plot$maeplot

external_slopes <- plot.model.slopeboxes(external,"External syndrome score",psych_scores[,c("sub","external_child")])
#external_slopes
#####

# TOTAL PSYCHOPATHOLOGY
#####
colnames(total) <- gsub("_child","",colnames(total))
total_uw <- dplyr::select(total,!contains("_weighted"))
total_plot <- plot.maes.slopes(total_uw,"Total syndrome score",
                                  "total syndrome score",psych_scores[,c("sub","total_child")])
#total_plot$maeplot

total_slopes <- plot.model.slopeboxes(total,"Total syndrome score",psych_scores[,c("sub","total_child")])
#total_slopes
#####

# AGE
#####
age_uw <- dplyr::select(age,!contains("_weighted"))
age_plot <- plot.maes.slopes(age_uw,"Brain age","brain_age",age_scores)
#age_plot$maeplot

age_slopes <- plot.model.slopeboxes(age,"Brain age",age_scores)
#age_slopes
#####

# WEIGHTED/UNWEIGHTED COMPARISONS MAE
# (Supplement Figure 1: Effects of sample weights on model errors)
#####
ccorr_comp <- plot.maecompboxes(ccorr,"Corrected crystallized cognition","corrected crystallized cognition",cogcorr_scores[,c("sub","crystallized_corr")])
cucorr_comp <- plot.maecompboxes(cucorr,"Uncorrected crystallized cognition","uncorrected crystallized cognition",cogucorr_scores[,c("sub","crystallized_uncorr")])
fcorr_comp <- plot.maecompboxes(fcorr,"Corrected fluid cognition","corrected fluid cognition",cogcorr_scores[,c("sub","fluid_corr")])
fucorr_comp <- plot.maecompboxes(fucorr,"Uncorrected fluid cognition","uncorrected fluid cognition",cogucorr_scores[,c("sub","fluid_uncorr")])
tcorr_comp <- plot.maecompboxes(tcorr,"Corrected total cognition","corrected total cognition",cogcorr_scores[,c("sub","tot_cognition_corr")])
tucorr_comp <- plot.maecompboxes(tucorr,"Uncorrected total cognition","uncorrected total cognition",cogucorr_scores[,c("sub","tot_cognition_uncorr")])

int_comp <- plot.maecompboxes(internal,"Internal syndrome scores","internal syndrome scores",psych_scores[,c("sub","internal_child")])
ext_comp <- plot.maecompboxes(external,"External syndrome scores","external syndrome scores",psych_scores[,c("sub","external_child")])
totpsy_comp <- plot.maecompboxes(total,"Total syndrome scores","total syndrome scores",psych_scores[,c("sub","total_child")])
age_comp <- plot.maecompboxes(age,"Brain age","brain age",age_scores)
#####

# PLOTTING FEATURE IMPORTANCES (by submod, function)
fi.submodplots <- function(submod_df,submod_name) {
  fis.long <- melt(submod_df,id.vars=c("features"))
  fis.long$variable <- sub(paste0(submod_name,"_"),"",fis.long$variable)
  fis.long$network <- gordon333networknames[gordon333partition]
  features_by_network <- ggplot(data=fis.long,aes(y=network,x=value,fill=variable,color=variable)) +
    geom_bar(stat="summary",position="dodge",fun="mean",color="black") +
    scale_fill_brewer(palette="Set1") +
    scale_color_brewer(palette="Set1") +
    # geom_jitter(stat="identity",size=0.45) +
    theme_classic()
  return(features_by_network)
  #return(fis.long)
}

# PLOTTING BY SUBMODALITY BY NETWORK (using function/UNFINISHED)
# (Supplement Figure 2: Feature importance by network by outcome per feature, most granular)
#####
crystallized_corr <- fi.submodplots(cogcorr_features %>% dplyr::select(c(features) | starts_with("crystallized_corr") & !contains("_weighted")),
                                    "crystallized_corr")
crystallized_uncorr <- fi.submodplots(cogucorr_features %>% dplyr::select(c(features) | starts_with("crystallized_uncorr") & !contains("_weighted")),
                                    "crystallized_uncorr")
fluid_corr <- fi.submodplots(cogcorr_features %>% dplyr::select(c(features) | starts_with("fluid_corr") & !contains("_weighted")),
                                    "fluid_corr")
fluid_uncorr <- fi.submodplots(cogucorr_features %>% dplyr::select(c(features) | starts_with("fluid_uncorr") & !contains("_weighted")),
                                    "fluid_uncorr")
total_corr <- fi.submodplots(cogcorr_features %>% dplyr::select(c(features) | starts_with("tot_cognition_corr") & !contains("_weighted")),
                                    "tot_cognition_corr")
total_uncorr <- fi.submodplots(cogucorr_features %>% dplyr::select(c(features) | starts_with("tot_cognition_uncorr") & !contains("_weighted")),
                                    "tot_cognition_uncorr")

# internal
# external
# total
# age

#####

# PLOTTING BY MEASURE (ACROSS MODALITIES) BY NETWORK (function)
fi.crossmod.barplots <- function(measure_df,measure_name, modality_choice) {
  measure.long <- melt(measure_df,id.vars=c("features"))
  measure.long$variable <- as.character(measure.long$variable)
  measure.long <- measure.long %>% mutate(feature = sub("^(.*)(_.*?)$","\\2",variable),
                                          variable = sub("^(.*)(_.*?)$","\\1",variable))
  measure.long <- measure.long %>% mutate(feature = sub("_","",feature))
  measure.long <- measure.long %>% filter(feature == measure_name)
  measure.long <- measure.long %>% mutate(modality = case_when(
    variable %in% cogcorr_vars ~ "Corrected cognition",
    variable %in% cogucorr_vars ~ "Uncorrected cognition",
    variable %in% psych_vars ~ "Psychopathology",
    variable == "age" ~ "Age"
  ))

  measure.long <- measure.long %>% filter(modality == modality_choice)
  measure.long$network <- gordon333networknames[gordon333partition]
  measure_plot <- ggplot(measure.long,aes(x=value,y=network,fill=variable,color=variable)) +
    geom_bar(stat="summary",position="dodge",fun="mean",color="black") +
    scale_fill_brewer(palette="Set1") +
    scale_color_brewer(palette="Set1") +
    geom_jitter(stat="identity",size=0.45) +
    theme_classic()

  return(measure_plot)
}

fi.allmod.barplots <- function(measure_df,measure_name) {
  measure_df <- measure_df %>% dplyr::select(c(features) | contains(measure_name))
  colnames(measure_df) <- sub("^(.*)(_.*?)$","\\1",colnames(measure_df))
  measure_df$cogcorr <- rowMeans(measure_df[,c(cogcorr_vars)])
  measure_df$cogucorr <- rowMeans(measure_df[,c(cogucorr_vars)])
  measure_df$psych <- rowMeans(measure_df[,c(psych_vars)])
  measure_df <- measure_df %>% dplyr::select(c(features,cogcorr,cogucorr,psych,age))
  measure.long <- melt(measure_df,id.vars=c("features"))
  measure.long$variable <- as.character(measure.long$variable)
  measure.long$network <- gordon333networknames[gordon333partition]
  
  measure_plot <- ggplot(measure.long,aes(y=network,x=value,fill=variable)) +
    geom_bar(stat="summary",position="dodge",fun="mean",color="black") +
    scale_fill_brewer(palette="Set1",name="Modality",labels=c("Age","Corrected cognition","Uncorrected cognition","Psychopathology")) +
    #scale_color_brewer(palette="Set1",name="Modality",labels=c("Age","Corrected cognition","Uncorrected cognition","Psychopathology")) +
    #geom_jitter(stat="identity",size=0.75) +
    theme_classic() +
    scale_y_continuous(limits=c(0,0.007))
  return(measure_plot)
}

# DATA WRANGLING CROSS-MODAL FEATURE IMPORTANCES
#####
cog.features <- merge(cogcorr_features,cogucorr_features,by="features")
cogpsych.features <- merge(cog.features,psych_features,by="features")
all.features <- merge(cogpsych.features,age_features,by="features")

nw.features <- all.features %>% dplyr::select(!contains("_weighted"))
# weighted features might not be useful to include(?) - weighted features only in one section
#w.features <- all.features %>% dplyr::select(c(features) | contains("_weighted"))
#####


# PLOTS OF FEATURE IMPORTANCE BY MEASURE BY MODALITY/SUBMODALITY
# (Supplemental Figure 3: Feature importance by measure by network per submodality)
# (Figure 2: Feature importance by measure by modality - evidence of differences in SF2 & SF3)
#####
#nw_alff_plot <- fi.crossmod.barplots(nw.features,"alff","Psychopathology")
#nw_alff_plot <- fi.crossmod.barplots(nw.features,"alff","Corrected cognition")

nw_alff_plot <- fi.allmod.barplots(nw.features,"alff")
nw_bcentr_plot <- fi.allmod.barplots(nw.features,"bcentr")
nw_clust_plot <- fi.allmod.barplots(nw.features,"clust")
nw_eff_plot <- fi.allmod.barplots(nw.features,"eff")
nw_intrafc_plot <- fi.allmod.barplots(nw.features,"intrafc")
nw_interfc_plot <- fi.allmod.barplots(nw.features,"interfc")
nw_part_plot <- fi.allmod.barplots(nw.features,"part")
nw_reho_plot <- fi.allmod.barplots(nw.features,"reho")
nw_str_plot <- fi.allmod.barplots(nw.features,"str")
#####

# Figure arrangement (using ggarrange/UNFINISHED)
#####
fig1abc <- ggarrange(cryst_corr$maeplot,
                    fluid_corr$maeplot,
                    total_corr$maeplot,
                    nrow = 1,common.legend = T,
                    legend = c("top"),
                    labels = "auto")
fig1def <- ggarrange(cryst_uncorr$maeplot,
                     fluid_uncorr$maeplot,
                     total_uncorr$maeplot,
                     nrow = 1,common.legend = T,
                     legend = "none",
                     labels = c("d","e","f"))
fig1ghi <- ggarrange(internal_plot$maeplot,
                     external_plot$maeplot,
                     total_plot$maeplot,
                     nrow = 1,common.legend = T,
                     legend = "none",
                     labels = c("g","h","i"))
fig1j <- ggarrange(age_plot$maeplot,
                   nrow = 1,common.legend = T,
                   legend = "none",
                   labels = c("j"))

fig2abc <- ggarrange(cryst_corrslopes,
                     fluid_corrslopes,
                     total_corrslopes,
                     nrow = 1, common.legend = T,
                     legend = c("top"),
                     labels = "auto")
fig2def <- ggarrange(cryst_uncorrslopes,
                     fluid_uncorrslopes,
                     total_uncorrslopes,
                     nrow = 1, common.legend = T,
                     legend = "none",
                     labels = c("d","e","f"))
fig2ghi <- ggarrange(internal_slopes,
                     external_slopes,
                     total_slopes,
                     nrow = 1, common.legend = T,
                     legend = "none",
                     labels = c("g","h","i"))
fig2j <- ggarrange(age_slopes,
                   nrow = 1, common.legend = T,
                   legend = "none",
                   labels = c("j"))

fig3abc <- ggarrange(nw_alff_plot + rremove("ylab") + rremove("xlab") + ggtitle("ALFF"),
                     nw_bcentr_plot + rremove("ylab") + rremove("xlab") + ggtitle("Betweenness centrality"),
                     nw_str_plot + rremove("ylab") + rremove("xlab") + ggtitle("Node strength"),
                     nrow=1,common.legend = T,legend=c("top"),
                     labels="auto")
annotate_figure(fig3abc, left = textGrob("Network", rot = 90, vjust = 1, gp = gpar(cex = 1)),
                bottom = textGrob("Mean decrease in impurity", gp = gpar(cex = 1)))

fig3def <- ggarrange(nw_eff_plot + rremove("ylab") + rremove("xlab") + ggtitle("Nodal efficiency"),
                     nw_clust_plot + rremove("ylab") + rremove("xlab") + ggtitle("Clustering coefficient"),
                     nw_part_plot + rremove("ylab") + rremove("xlab") + ggtitle("Participation coefficient"),
                     nrow=1,common.legend = T,legend="none",
                     labels=c("d","e","f"))
annotate_figure(fig3def, left = textGrob("Network", rot = 90, vjust = 1, gp = gpar(cex = 1)),
                bottom = textGrob("Mean decrease in impurity", gp = gpar(cex = 1)))

fig3ghi <- ggarrange(nw_intrafc_plot + rremove("ylab") + rremove("xlab") + ggtitle("Within-network FC"),
                     nw_interfc_plot + rremove("ylab") + rremove("xlab") + ggtitle("Between-network FC"),
                     nw_reho_plot + rremove("ylab") + rremove("xlab") + ggtitle("Regional homogeneity"),
                     nrow=1,common.legend = T,legend="none",
                     labels=c("g","h","i"))
annotate_figure(fig3ghi, left = textGrob("Network", rot = 90, vjust = 1, gp = gpar(cex = 1)),
                bottom = textGrob("Mean decrease in impurity", gp = gpar(cex = 1)))

#####

