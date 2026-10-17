# N2 第二步：LASSO（10 折 CV，λ.1se/λ.min）与 Boruta 特征筛选
# 输入：n2_train.csv（MICE 插补后，仅训练集）
# 输出：n2_lasso.csv、n2_boruta.csv、n2_lasso_cv.png、n2_lasso_path.png、n2_boruta.png
suppressPackageStartupMessages({
  library(glmnet); library(Boruta); library(randomForest)
})
set.seed(123)
OUT <- Sys.getenv("CN2_OUT", ".")
d <- read.csv(file.path(OUT, "n2_train.csv"), check.names = FALSE, fileEncoding = "UTF-8-BOM")
names(d)[1] <- sub("^\ufeff", "", names(d)[1])
FEATS <- c("bun_post_max","wbc_post_max","hct_post_min","hb_post_min","plt_post_min","mcv_post_max",
           "mch_post_max","mchc_post_max","rdw_post_max","na_post_min","k_post_max","cl_post_min",
           "ca_post_min","hco3_post_min","agap_post_max","glu_post_max","hr_post_max","rr_post_max",
           "spo2_post_min","sbp_post_min","dbp_post_min","mbp_post_min","bun_cr_ratio","shock_index_max",
           "age","female","sofa","cr_base","vent","vaso","xoi_base","htn","dm","chf","ckd","lung",
           "mi","cva","cancer","liver","gout")
miss <- setdiff(FEATS, names(d))
if (length(miss)) { cat("缺列:", paste(miss, collapse=", "), "\n"); quit(status=1) }
X <- as.matrix(d[, FEATS]); y <- d$y
cat("训练集:", nrow(X), "例；事件:", sum(y), "；候选变量:", length(FEATS), "\n")

## ---------------- LASSO ----------------
cvfit <- cv.glmnet(X, y, family = "binomial", alpha = 1, standardize = TRUE,
                   type.measure = "auc", nfolds = 10, seed = 123)
co_1se <- as.matrix(coef(cvfit, s = "lambda.1se"))
co_min <- as.matrix(coef(cvfit, s = "lambda.min"))
lasso <- data.frame(var = rownames(co_1se), coef_1se = as.numeric(co_1se), coef_min = as.numeric(co_min),
                    row.names = NULL)
lasso$lasso_1se <- lasso$coef_1se != 0
lasso$lasso_min <- lasso$coef_min != 0
lasso <- lasso[lasso$var != "(Intercept)", ]
lasso <- lasso[order(-abs(lasso$coef_1se)), ]
write.csv(lasso, file.path(OUT, "n2_lasso.csv"), row.names = FALSE, fileEncoding = "UTF-8")
cat("\nLASSO λ.1se 选中:", sum(lasso$lasso_1se), "个；λ.min 选中:", sum(lasso$lasso_min), "个\n")
cat("λ.1se 选中变量:", paste(lasso$var[lasso$lasso_1se], collapse = ", "), "\n")
cat("CV 最优 AUC:", round(max(cvfit$cvm), 4), "  λ.1se AUC:", round(cvfit$cvm[which(cvfit$lambda == cvfit$lambda.1se)], 4), "\n")

png(file.path(OUT, "n2_lasso_cv.png"), width = 1600, height = 1300, res = 200)
par(mar = c(4.5, 4.5, 3, 1))
plot(cvfit); title(main = "LASSO 10 折交叉验证（AUC）", line = 2.2)
dev.off()

png(file.path(OUT, "n2_lasso_path.png"), width = 1700, height = 1400, res = 200)
par(mar = c(4.5, 4.5, 3, 1))
fit_all <- glmnet(X, y, family = "binomial", alpha = 1, standardize = TRUE)
plot(fit_all, xvar = "lambda", label = TRUE, lwd = 1.1)
abline(v = log(cvfit$lambda.1se), lty = 2, col = "red"); abline(v = log(cvfit$lambda.min), lty = 3, col = "blue")
title(main = "LASSO 系数路径（红=λ.1se，蓝=λ.min）", line = 2.2)
dev.off()

## ---------------- Boruta ----------------
bd <- d[, c(FEATS, "y")]
bd$y <- factor(bd$y)
bor <- Boruta(y ~ ., data = bd, doTrace = 0, maxRuns = 100, ntree = 300)
bt <- attStats(bor)
bt$var <- rownames(bt)
bo <- data.frame(var = bt$var, decision = bt$decision, medianImp = bt$medianImp,
                 meanImp = bt$meanImp, minImp = bt$minImp, maxImp = bt$maxImp, normHits = bt$normHits,
                 row.names = NULL)
bo <- bo[order(-bo$medianImp), ]
write.csv(bo, file.path(OUT, "n2_boruta.csv"), row.names = FALSE, fileEncoding = "UTF-8")
cat("\nBoruta：Confirmed", sum(bo$decision == "Confirmed"), "／Tentative",
    sum(bo$decision == "Tentative"), "／Rejected", sum(bo$decision == "Rejected"), "\n")
cat("Confirmed 变量:", paste(bo$var[bo$decision == "Confirmed"], collapse = ", "), "\n")

png(file.path(OUT, "n2_boruta.png"), width = 1700, height = 1400, res = 200)
par(mar = c(12, 4.5, 3, 1))
plot(bor, las = 2, cex.axis = 0.8, xlab = "")
title(main = "Boruta 特征重要性（绿色=Confirmed，黄色=Tentative，红色=Rejected）", line = 2.2)
dev.off()
cat("\nSAVED: n2_lasso.csv / n2_boruta.csv / n2_lasso_cv.png / n2_lasso_path.png / n2_boruta.png\n")
