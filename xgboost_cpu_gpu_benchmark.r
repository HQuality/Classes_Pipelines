
library(xgboost)

set.seed(1)
N <- 10000000
p <- 100
pp <- 25
X <- matrix(runif(N * p), ncol = p)
betas <- 2 * runif(pp) - 1
sel <- sort(sample(p, pp))
m <- X[, sel] %*% betas - 1 + rnorm(N)
y <- rbinom(N, 1, plogis(m))

tr <- sample.int(N, N * 0.90)

trainer <- function(n_cpus, n_gpus, n_iterations, n_depth) {
  
  if (n_gpus == 0) {
    
    pt <- proc.time()
    set.seed(11111)
    model <- xgb.train(list(objective = "binary:logistic", eval_metric = "logloss", nthread = n_cpus, eta = 0.10, max_depth = n_depth, max_bin = 64, tree_method = "hist"),
                       dtrain, watchlist = wl, nrounds = n_iterations, verbose = 0)
    my_time <- proc.time() - pt
    
  } else {
    
    pt <- proc.time()
    set.seed(11111)
    model <- xgb.train(list(objective = "gpu:binary:logistic", eval_metric = "logloss", nthread = n_cpus, eta = 0.10, max_depth = n_depth, max_bin = 64, tree_method = "gpu_hist", n_gpus = n_gpus),
                       dtrain, watchlist = wl, nrounds = n_iterations, verbose = 0)
    my_time <- proc.time() - pt
    
  }
  
  rm(model)
  gc(verbose = FALSE)
  
  return(my_time)
  
}

dtrain <- xgb.DMatrix(X[tr,], label = y[tr])
dtest <- xgb.DMatrix(X[-tr,], label = y[-tr])
wl <- list(test = dtest)

library(data.table)

log_file <- "/home/laurae/Documents/R/xgboost_GPU_test/log.csv"
data_file <- "/home/laurae/Documents/R/xgboost_GPU_test/data.csv"
n_cpus <- c(1, 2, 4, 6, 9, 17, 18, 19, 35, 36, 37, 70, 71, 72)
n_cpus_rep <- 2
n_gpus <- c(1, 2, 3, 4)
n_gpus_rep <- 4
n_iters <- 500
n_depths <- 2:12

data <- rbindlist(list(data.table(Workload = "GPU", CPU = rep(rev(n_gpus), each = length(n_depths)), GPU = rep(rev(n_gpus), each = length(n_depths)), Depth = rep(rev(n_depths), length(n_gpus)), Repeats = n_gpus_rep, Speed = 0),
                       data.table(Workload = "CPU", CPU = rep(rev(n_cpus), each = length(n_depths)), GPU = 0, Depth = rep(rev(n_depths), length(n_cpus)), Repeats = n_cpus_rep, Speed = 0)))
colnames(data) <- c("Workload", "CPU Threads", "GPU Count", "Depth", "Repeats", "Speed")

sink(log_file, append = FALSE)

for (k in seq_len(nrow(data))) {
  
  cat(k, "\n", sep = "", file = log_file, append = TRUE)
  
  if (k > 1) {
    if ((data$`GPU Count`[k - 1] - data$`GPU Count`[k]) != 0) {
      rm(dtrain, dtest, wl)
      gc(verbose = FALSE)
      dtrain <- xgb.DMatrix(X[tr,], label = y[tr])
      dtest <- xgb.DMatrix(X[-tr,], label = y[-tr])
      wl <- list(test = dtest)
      gc(verbose = FALSE)
    }
  }
  
  zz <- 0
  z <- 0
  
  for (j in seq_len(data$Repeats[k])) {
    
    zz <- unname(trainer(data$`CPU Threads`[k], data$`GPU Count`[k], n_iters, data$Depth[k]))[3]
    z <- z + zz
    
    cat("[", sprintf(paste0("%0", floor(log10(max(data$Repeats)) + 1), "d"), j), " / ", data$Repeats[k], "] ", sprintf("%02d", data$`CPU Threads`[k]), " CPUs / ", data$`GPU Count`[k], " GPUs / ", sprintf("%02d", data$Depth[k]), " Depth: ", sprintf("%08.03f", zz), "s\n", sep = "", file = log_file, append = TRUE)
    gc(verbose = FALSE)
    
  }
  
  data[k, Speed := z / j]
  cat("[Done] ", sprintf("%02d", data$`CPU Threads`[k]), " CPUs / ", data$`GPU Count`[k], " GPUs / ", sprintf("%02d", data$Depth[k]), " Depth: ", sprintf("%08.03f", data$Speed[k]), "s\n\n", sep = "", file = log_file, append = TRUE)
  
}

sink()

fwrite(data, data_file)