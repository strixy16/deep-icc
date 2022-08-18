library("survival")
library("survminer")
library("survAUC")
library("readxl")
library("writexl")

# Load in data spreadsheet for all data
full_TR_liver <- read_excel("/Users/katyscott/Desktop/HDFS_Project/Data/FeatureSelection/HCC_MCRC_ICC_HDFS_80_20/train_liver_feats_and_labels.xlsx")
full_TE_liver <- read_excel("/Users/katyscott/Desktop/HDFS_Project/Data/FeatureSelection/HCC_MCRC_ICC_HDFS_80_20/test_liver_feats_and_labels.xlsx")

full_TR_tumor <- read_excel("/Users/katyscott/Desktop/HDFS_Project/Data/FeatureSelection/HCC_MCRC_ICC_HDFS_80_20/train_tumor_feats_and_labels.xlsx")
full_TE_tumor <- read_excel("/Users/katyscott/Desktop/HDFS_Project/Data/FeatureSelection/HCC_MCRC_ICC_HDFS_80_20/test_tumor_feats_and_labels.xlsx")

full_TR_livertumor <- read_excel("/Users/katyscott/Desktop/HDFS_Project/Data/FeatureSelection/HCC_MCRC_ICC_HDFS_80_20/train_liver_tumor_feats_and_labels.xlsx")
full_TE_livertumor <- read_excel("/Users/katyscott/Desktop/HDFS_Project/Data/FeatureSelection/HCC_MCRC_ICC_HDFS_80_20/test_liver_tumor_feats_and_labels.xlsx")

TR_liver <- subset(full_TR_liver, select = -c(ScoutID, Cancer_Type))
TE_liver <- subset(full_TE_liver, select = -c(ScoutID, Cancer_Type))
  
TR_tumor <- subset(full_TR_tumor, select = -c(ScoutID, Cancer_Type))
TE_tumor <- subset(full_TE_tumor, select = -c(ScoutID, Cancer_Type))

TR_livertumor <- subset(full_TR_livertumor, select = -c(ScoutID, Cancer_Type))
TE_livertumor <- subset(full_TE_livertumor, select = -c(ScoutID, Cancer_Type))



# Set up CoxPH for liver data
liver.fit <- coxph(Surv(HDFS_Time, HDFS_Code) ~ Kurtosis+MeanAbsoluteDeviation+
                     Range+Variance+Busyness+Complexity+Contrast+Strength+
                     Maximum2DDiameterRow+MinorAxisLength+SurfaceVolumeRatio,
                   x=TRUE, y=TRUE, method="breslow", data=TR_liver)
# Check proportional hazards
liver.ph <- cox.zph(liver.fit)
liver.ph
# Predict for test data
liverpred <- predict(liver.fit, newdata=TE_liver)

GHCI(liverpred)

survConcordance(formula = Surv(HDFS_Time, HDFS_Code) ~ predict(liver.fit, TE_liver), TE_liver)


# Set up CoxPH for tumor data
tumor.fit <- coxph(Surv(HDFS_Time, HDFS_Code) ~ Entropy+InterquartileRange+
                     Kurtosis+Minimum+Range+RobustMeanAbsoluteDeviation+Complexity+Sphericity,
                   x=TRUE, y=TRUE, method="breslow", data=TR_tumor)
# Check proportional hazards
tumor.ph <- cox.zph(tumor.fit)
tumor.ph 

# Predict for test data
tumorpred <- predict(tumor.fit, newdata=TE_tumor)

GHCI(tumorpred)
survConcordance(formula = Surv(HDFS_Time, HDFS_Code) ~ predict(tumor.fit, TE_tumor), TE_tumor)



# Set up CoxPH for liver and tumor data
livertumor.fit <- coxph(Surv(HDFS_Time, HDFS_Code) ~ Kurtosis+
                          Range+Variance+Busyness+Complexity+Contrast+Strength+Elongation+
                          Maximum2DDiameterRow+MinorAxisLength,
                        x=TRUE, y=TRUE, method="breslow", data=TR_livertumor)
livertumor.ph <- cox.zph(livertumor.fit)
liver.ph

livertumorpred <- predict(livertumor.fit, newdata=TE_livertumor)

GHCI(livertumorpred)
survConcordance(formula = Surv(HDFS_Time, HDFS_Code) ~ predict(livertumor.fit, TE_livertumor), TE_livertumor)

tbl_livertumorpred = full_TE_livertumor[c('ScoutID', 'HDFS_Time', 'HDFS_Code', 'Cancer_Type')]
tbl_tumorpred = full_TE_tumor[c('ScoutID', 'HDFS_Time', 'HDFS_Code', 'Cancer_Type')]
tbl_liverpred = full_TE_liver[c('ScoutID', 'HDFS_Time', 'HDFS_Code', 'Cancer_Type')]

tbl_livertumorpred['Prediction'] = as.data.frame(livertumorpred)
tbl_tumorpred['Prediction'] = as.data.frame(tumorpred)
tbl_liverpred['Prediction'] = as.data.frame(liverpred)

write_xlsx(tbl_livertumorpred, "/Users/katyscott/Desktop/HDFS_Project/Data/FeatureSelection/HCC_MCRC_ICC_HDFS_80_20/CPH_test_liver_tumor_predictions.xlsx")
write_xlsx(tbl_tumorpred, "/Users/katyscott/Desktop/HDFS_Project/Data/FeatureSelection/HCC_MCRC_ICC_HDFS_80_20/CPH_test_tumor_predictions.xlsx")
write_xlsx(tbl_liverpred, "/Users/katyscott/Desktop/HDFS_Project/Data/FeatureSelection/HCC_MCRC_ICC_HDFS_80_20/CPH_test_liver_predictions.xlsx")


