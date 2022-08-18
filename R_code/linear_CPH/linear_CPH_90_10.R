library("survival")
library("survminer")
library("survAUC")
library("readxl")
library("writexl")

# Load in data spreadsheet for all data
full_TR_liver <- read_excel("/Users/katyscott/Desktop/HDFS_Project/Data/FeatureSelection/HCC_MCRC_ICC_HDFS_90_10/train_liver_feats_and_labels.xlsx")
full_TE_liver <- read_excel("/Users/katyscott/Desktop/HDFS_Project/Data/FeatureSelection/HCC_MCRC_ICC_HDFS_90_10/test_liver_feats_and_labels.xlsx")

full_TR_tumor <- read_excel("/Users/katyscott/Desktop/HDFS_Project/Data/FeatureSelection/HCC_MCRC_ICC_HDFS_90_10/train_tumor_feats_and_labels.xlsx")
full_TE_tumor <- read_excel("/Users/katyscott/Desktop/HDFS_Project/Data/FeatureSelection/HCC_MCRC_ICC_HDFS_90_10/test_tumor_feats_and_labels.xlsx")

# normalize Energy variable so it can be used in tumor model
#norm_TR_tumor_energy = full_TR_tumor$Energy / (mean(full_TR_tumor$Energy))
#norm_TE_tumor_energy = full_TE_tumor$Energy / (mean(full_TE_tumor$Energy))
#full_TR_tumor['norm_Energy'] = as.data.frame(norm_TR_tumor_energy)
#full_TE_tumor['norm_Energy'] = as.data.frame(norm_TE_tumor_energy)

full_TR_livertumor <- read_excel("/Users/katyscott/Desktop/HDFS_Project/Data/FeatureSelection/HCC_MCRC_ICC_HDFS_90_10/train_liver_tumor_feats_and_labels.xlsx")
full_TE_livertumor <- read_excel("/Users/katyscott/Desktop/HDFS_Project/Data/FeatureSelection/HCC_MCRC_ICC_HDFS_90_10/test_liver_tumor_feats_and_labels.xlsx")

# normalize Surface Area variable so it can be 

TR_liver <- subset(full_TR_liver, select = -c(ScoutID, Cancer_Type))
TE_liver <- subset(full_TE_liver, select = -c(ScoutID, Cancer_Type))
  
TR_tumor <- subset(full_TR_tumor, select = -c(ScoutID, Cancer_Type))
TE_tumor <- subset(full_TE_tumor, select = -c(ScoutID, Cancer_Type))

TR_livertumor <- subset(full_TR_livertumor, select = -c(ScoutID, Cancer_Type))
TE_livertumor <- subset(full_TE_livertumor, select = -c(ScoutID, Cancer_Type))


# Set up CoxPH for liver data
liver.fit <- coxph(Surv(HDFS_Time, HDFS_Code) ~ Entropy+Kurtosis+Minimum+Skewness+Uniformity+
                     Variance+Busyness+Complexity+Strength+Maximum2DDiameterColumn+
                     MinorAxisLength+SurfaceVolumeRatio,
                   x=TRUE, y=TRUE, method="breslow", data=TR_liver)
# Check proportional hazards
liver.ph <- cox.zph(liver.fit)
#liver.ph

livertrainpred <- predict(liver.fit, newdata=TR_liver)
livertestpred <- predict(liver.fit, newdata=TE_liver)

#summary(liver.fit)
#GHCI(livertrainpred)
#GHCI(livertestpred)
survConcordance(formula = Surv(HDFS_Time, HDFS_Code) ~ predict(liver.fit, TE_liver), TE_liver)

liverSurv.rsp <- Surv(TR_liver$HDFS_Time, TR_liver$HDFS_Code)
liverSurv.rsp.new <- Surv(TE_liver$HDFS_Time, TE_liver$HDFS_Code)
livertrainCstat <- UnoC(liverSurv.rsp, liverSurv.rsp, livertrainpred)
livertestCstat <- UnoC(liverSurv.rsp, liverSurv.rsp.new, livertestpred)


# Set up CoxPH for tumor data
tumor.fit <- coxph(Surv(HDFS_Time, HDFS_Code) ~ 
                     Kurtosis+MeanAbsoluteDeviation+
                     Minimum+RobustMeanAbsoluteDeviation+Skewness+Uniformity+Variance+
                     Complexity+Sphericity,
                   x=TRUE, y=TRUE, method="breslow", data=TR_tumor)
# Check proportional hazards
tumor.ph <- cox.zph(tumor.fit)
#tumor.ph 

tumortrainpred <- predict(tumor.fit, newdata=TR_tumor)
tumortestpred <- predict(tumor.fit, newdata=TE_tumor)

#summary(tumor.fit)
#GHCI(tumortrainpred)
#GHCI(tumortestpred)
survConcordance(formula = Surv(HDFS_Time, HDFS_Code) ~ predict(tumor.fit, TE_tumor), TE_tumor)

tumorSurv.rsp <- Surv(TR_tumor$HDFS_Time, TR_tumor$HDFS_Code)
tumorSurv.rsp.new <- Surv(TE_tumor$HDFS_Time, TE_tumor$HDFS_Code)
tumortrainCstat <- UnoC(tumorSurv.rsp, tumorSurv.rsp, tumortrainpred)
tumortestCstat <- UnoC(tumorSurv.rsp, tumorSurv.rsp.new, tumortestpred)


# Set up CoxPH for liver and tumor data
livertumor.fit <- coxph(Surv(HDFS_Time, HDFS_Code) ~ Range+Variance+
                          Busyness+Complexity+Contrast+Strength+Elongation+
                          Maximum2DDiameterRow+MinorAxisLength,
                        x=TRUE, y=TRUE, method="breslow", data=TR_livertumor)
livertumor.ph <- cox.zph(livertumor.fit)
#livertumor.ph

livertumortrainpred <- predict(livertumor.fit, newdata=TR_livertumor)
livertumortestpred <- predict(livertumor.fit, newdata=TE_livertumor)

#summary(livertumor.fit)
#GHCI(livertumortrainpred)
#GHCI(livertumortestpred)
survConcordance(formula = Surv(HDFS_Time, HDFS_Code) ~ predict(livertumor.fit, TE_livertumor), TE_livertumor)

livertumorSurv.rsp <- Surv(TR_livertumor$HDFS_Time, TR_livertumor$HDFS_Code)
livertumorSurv.rsp.new <- Surv(TE_livertumor$HDFS_Time, TE_livertumor$HDFS_Code)
livertumortrainCstat <- UnoC(livertumorSurv.rsp, livertumorSurv.rsp, livertumortrainpred)
livertumortestCstat <- UnoC(livertumorSurv.rsp, livertumorSurv.rsp.new, livertumortestpred)


tbl_livertumorpred = full_TE_livertumor[c('ScoutID', 'HDFS_Time', 'HDFS_Code', 'Cancer_Type')]
tbl_tumorpred = full_TE_tumor[c('ScoutID', 'HDFS_Time', 'HDFS_Code', 'Cancer_Type')]
tbl_liverpred = full_TE_liver[c('ScoutID', 'HDFS_Time', 'HDFS_Code', 'Cancer_Type')]

tbl_livertumorpred['Prediction'] = as.data.frame(livertumortestpred)
tbl_tumorpred['Prediction'] = as.data.frame(tumortestpred)
tbl_liverpred['Prediction'] = as.data.frame(livertestpred)

#write_xlsx(tbl_livertumorpred, "/Users/katyscott/Desktop/HDFS_Project/Data/FeatureSelection/HCC_MCRC_ICC_HDFS_90_10/CPH_test_liver_tumor_predictions.xlsx")
#write_xlsx(tbl_tumorpred, "/Users/katyscott/Desktop/HDFS_Project/Data/FeatureSelection/HCC_MCRC_ICC_HDFS_90_10/CPH_test_tumor_predictions.xlsx")
#write_xlsx(tbl_liverpred, "/Users/katyscott/Desktop/HDFS_Project/Data/FeatureSelection/HCC_MCRC_ICC_HDFS_90_10/CPH_test_liver_predictions.xlsx")

num_samples <- nrow(TE_liver)

df_liver_confidence <- data.frame(x1 = numeric(0))
df_tumor_confidence <- data.frame(x1 = numeric(0))
df_livertumor_confidence <- data.frame(x1 = numeric(0))
colnames(df_liver_confidence) <- c('UnoC')
colnames(df_tumor_confidence) <- c('UnoC')
colnames(df_livertumor_confidence) <- c('UnoC')

for (x in 1:1000){
  
  samp_liver <- TE_liver[sample(nrow(TE_liver), nrow(TE_liver), replace=TRUE),]
  samp_tumor <- TE_tumor[sample(nrow(TE_tumor), nrow(TE_tumor), replace=TRUE),]
  samp_livertumor <- TE_liver[sample(nrow(TE_livertumor), nrow(TE_livertumor), replace=TRUE),]

  samp_liver_pred <- predict(liver.fit, newdata=samp_liver)
  samp_tumor_pred <- predict(tumor.fit, newdata=samp_tumor)
  samp_livertumor_pred <- predict(livertumor.fit, newdata=samp_livertumor)
  
  sampliverSurv <- Surv(samp_liver$HDFS_Time, samp_liver$HDFS_Code)
  samptumorSurv <- Surv(samp_tumor$HDFS_Time, samp_tumor$HDFS_Code)
  samplivertumorSurv <- Surv(samp_livertumor$HDFS_Time, samp_livertumor$HDFS_Code)
  
  sampliverCstat <- UnoC(liverSurv.rsp, sampliverSurv, samp_liver_pred)
  samptumorCstat <- UnoC(tumorSurv.rsp, samptumorSurv, samp_tumor_pred)
  samplivertumorCstat <- UnoC(livertumorSurv.rsp, samplivertumorSurv, samp_livertumor_pred)
  
  df_liver_confidence[x,] <- sampliverCstat
  df_tumor_confidence[x,] <- samptumorCstat
  df_livertumor_confidence[x,] <- samplivertumorCstat
}

write_xlsx(df_liver_confidence, "/Users/katyscott/Desktop/HDFS_Project/Data/FeatureSelection/HCC_MCRC_ICC_HDFS_90_10/CPH_liver_unoC_confidence.xlsx")
write_xlsx(df_tumor_confidence, "/Users/katyscott/Desktop/HDFS_Project/Data/FeatureSelection/HCC_MCRC_ICC_HDFS_90_10/CPH_tumor_unoC_confidence.xlsx")
write_xlsx(df_livertumor_confidence, "/Users/katyscott/Desktop/HDFS_Project/Data/FeatureSelection/HCC_MCRC_ICC_HDFS_90_10/CPH_livertumor_unoC_confidence.xlsx")


