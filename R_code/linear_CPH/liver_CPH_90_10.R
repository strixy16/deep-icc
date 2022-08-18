library("survival")
library("survminer")
library("survAUC")
library("readxl")
library("writexl")

# Load in data spreadsheet for all data
full_TR_liver <- read_excel("/Users/katyscott/Desktop/HDFS_Project/Data/FeatureSelection/HCC_MCRC_ICC_HDFS_90_10/train_liver_feats_and_labels.xlsx")
full_TE_liver <- read_excel("/Users/katyscott/Desktop/HDFS_Project/Data/FeatureSelection/HCC_MCRC_ICC_HDFS_90_10/test_liver_feats_and_labels.xlsx")

# normalize Surface Area variable so it can be 

TR_liver <- subset(full_TR_liver, select = -c(ScoutID, Cancer_Type))
TE_liver <- subset(full_TE_liver, select = -c(ScoutID, Cancer_Type))


# Set up CoxPH for liver data
liver.fit <- coxph(Surv(HDFS_Time, HDFS_Code) ~ Entropy+Kurtosis+Minimum+Skewness+Uniformity+
                     Variance+Busyness+Complexity+Strength+Maximum2DDiameterColumn+
                     MinorAxisLength+SurfaceVolumeRatio,
                   x=TRUE, y=TRUE, method="breslow", data=TR_liver)
# Check proportional hazards
liver.ph <- cox.zph(liver.fit)
liver.ph

livertrainpred <- predict(liver.fit, newdata=TR_liver)
livertestpred <- predict(liver.fit, newdata=TE_liver)

summary(liver.fit)
#GHCI(livertrainpred)
#GHCI(livertestpred)
survConcordance(formula = Surv(HDFS_Time, HDFS_Code) ~ predict(liver.fit, TE_liver), TE_liver)

liverSurv.rsp <- Surv(TR_liver$HDFS_Time, TR_liver$HDFS_Code)
liverSurv.rsp.new <- Surv(TE_liver$HDFS_Time, TE_liver$HDFS_Code)
livertrainCstat <- UnoC(liverSurv.rsp, liverSurv.rsp, livertrainpred)
livertestCstat <- UnoC(liverSurv.rsp, liverSurv.rsp.new, livertestpred)

# Cancer specific testing
HCC_rows <- full_TE_liver[full_TE_liver$Cancer_Type == 0,]
ICC_rows <- full_TE_liver[full_TE_liver$Cancer_Type == 2,]
MCRC_rows <- full_TE_liver[full_TE_liver$Cancer_Type == 1,]

HCC_liver <- subset(HCC_rows, select = -c(ScoutID, Cancer_Type))
ICC_liver <- subset(ICC_rows, select = -c(ScoutID, Cancer_Type))
MCRC_liver <- subset(MCRC_rows, select = -c(ScoutID, Cancer_Type))

liverHCCpred <- predict(liver.fit, newdata=HCC_liver)
liverHCC.rsp <- Surv(HCC_liver$HDFS_Time, HCC_liver$HDFS_Code)
liverHCCCstat <- UnoC(liverSurv.rsp, liverHCC.rsp, liverHCCpred)

liverICCpred <- predict(liver.fit, newdata=ICC_liver)
liverICC.rsp <- Surv(ICC_liver$HDFS_Time, ICC_liver$HDFS_Code)
liverICCCstat <- UnoC(liverSurv.rsp, liverICC.rsp, liverICCpred)

liverMCRCpred <- predict(liver.fit, newdata=MCRC_liver)
liverMCRC.rsp <- Surv(MCRC_liver$HDFS_Time, MCRC_liver$HDFS_Code)
liverMCRCCstat <- UnoC(liverSurv.rsp, liverMCRC.rsp, liverMCRCpred)

