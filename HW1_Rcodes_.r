# Reading the data
hw1_input <- read.csv("~/Desktop/IE582/HW1/hw1_files/hw1_input.csv")
hw1_real <- read.csv("~/Desktop/IE582/HW1/hw1_files/hw1_real.csv", header=FALSE)
hw1_img <- read.csv("~/Desktop/IE582/HW1/hw1_files/hw1_img.csv", header=FALSE)

# Installing libraries that probably will be needed
install.packages("ggplot2")
library(ggplot2)
install.packages("zoo")
library(zoo)
install.packages("quantmod")
library(quantmod)

# Editing the data - removing 1st row
hw1_real <- hw1_real[-1, ]
hw1_img <- hw1_img[-1, ]
rownames(hw1_real) <- NULL
rownames(hw1_img) <- NULL

# Printing structure and summary of data
str(hw1_input)
summary(hw1_input)

str(hw1_real)
str(hw1_img)

# Highest and lowest values of real component
all_values <- unlist(hw1_real) 
lowest_values <- sort(all_values, decreasing = FALSE)[1:5]
highest_values <- sort(all_values, decreasing = TRUE)[1:5]
print(lowest_values)
print(highest_values)

# Highest and lowest values of img component\
all_values2 <- unlist(hw1_img) 
lowest_values2 <- sort(all_values2, decreasing = FALSE)[1:5]
highest_values2 <- sort(all_values2, decreasing = TRUE)[1:5]
print(lowest_values2)
print(highest_values2)

# Checking missing values
colSums(is.na(hw1_input))

# Plotting first 6 rows of each dataset
data6_plot_real <- hw1_real[1:6, ]
matplot(t(data6_plot_real), type = "l", lty = 1, col = 1:6, xlab = "Frequencies", ylab = "Real Component", 
main = "First Six Rows of Data")

data6_plot_img <- hw1_img[1:6, ]
matplot(t(data6_plot_img), type = "l", lty = 1, col = 1:6, xlab = "Frequencies", ylab = "Img Component", 
main = "First Six Rows of Data")

magnitude_s11 <- sqrt(hw1_real^2+hw1_img^2)
data6_plot <- magnitude_s11[1:6, ]
matplot(t(data6_plot), type = "l", lty = 1, col = 1:6, xlab = "Frequencies", ylab = "S11 Values", main = "First Six Rows of Data")

#Checking the correlation between variables\
cor_matrix <- cor(hw1_input)
print(cor_matrix)

# PCA\
# Scaling the data before PCA
scaled_hw1_input <- scale(hw1_input)

# Checking variable behavior after scaling
pairs(scaled_hw1_input, main = "Scatter Plot Matrix of Scaled Features")
plot(scaled_hw1_input[, 1], scaled_hw1_input[, 2], 
      main = "Scatter Plot of Scaled Feature 1 vs. Feature 2",
      xlab = "Feature 1 (scaled)", 
      ylab = "Feature 2 (scaled)")

plot(scaled_hw1_input[, 2], scaled_hw1_input[, 3], 
            main = "Scatter Plot of Scaled Feature 2 vs. Feature 3",
            xlab = "Feature 2 (scaled)", 
            ylab = "Feature 3 (scaled)")

# Conducting PCA
pca_hw1_input <- princomp(scaled_hw1_input)
summary(pca_hw1_input)

# Visualizing variables
install.packages("factoextra")
library(factoextra)

fviz_pca_var(pca_hw1_input, col.var = "cos2",
              gradient.cols = c("black", "orange", "green"),
              repel = TRUE)

# Printing the loadings for first 8 PCs
pca_hw1_input$loadings[, 1:8]

# Plotting each components contribution on scree plot
plot(pca_hw1_input, type = "l", main = "Scree Plot")

# Barplot for PC1
barplot(pca_hw1_input$scores[,1])

# Lowest 10 S11 values in output dataset
magnitude_s11_long <- as.data.frame(as.table(as.matrix(magnitude_s11)))
colnames(magnitude_s11_long) <- c("Observation", "Frequency", "Value")
output_sorted <- magnitude_s11_long[order(magnitude_s11_long$Value), ]
lowest_10 <- head(output_sorted, 10)
print(lowest_10)

# Separating columns corresponding to three lowest frequencies
magnitude_s11_v72 <- magnitude_s11[ ,72]
magnitude_s11_v29 <- magnitude_s11[ ,29]
magnitude_s11_v28 <- magnitude_s11[ ,28]

# Making training and test data for V72 as output
set.seed(123) 
sample_index <- sample(1:nrow(hw1_input), 0.7 * nrow(hw1_input))
train_input <- hw1_input[sample_index, ]
train_output <- magnitude_s11_v72[sample_index]
test_input <- hw1_input[-sample_index, ]
test_output <- magnitude_s11_v72[-sample_index]
train_data <- data.frame(train_input, train_output)
test_data <- data.frame(test_input, test_output)
colnames(train_data)[ncol(train_data)] <- "S11_V72"

# Linear regression model for V72 as output
lm_v72_tt = lm(S11_V72~ length.of.patch + width.of.patch + height.of.patch + height.of.substrate + height.of.solder.resist.layer + radius.of.the.probe + c_pad + c_antipad + c_probe + dielectric.constant.of.substrate + dielectric.constant.of.solder.resist.layer, data = train_data)
summary(lm_v72_tt)

# Removing insignificant variables
lm_v72_tt2 = lm(S11_V72~ width.of.patch + height.of.substrate + height.of.solder.resist.layer + radius.of.the.probe + c_pad + c_probe + dielectric.constant.of.substrate, data = train_data)
summary(lm_v72_tt2)

# Making training and test data for V29 as output
set.seed(124) 
sample_index2 <- sample(1:nrow(hw1_input), 0.7 * nrow(hw1_input))
train_input2 <- hw1_input[sample_index2, ]
train_output2 <- magnitude_s11_v29[sample_index2]
test_input2 <- hw1_input[-sample_index2, ]
test_output2 <- magnitude_s11_v29[-sample_index2]
train_data2 <- data.frame(train_input2, train_output2)
test_data2 <- data.frame(test_input2, test_output2)
colnames(train_data2)[ncol(train_data2)] <- "S11_V29"

# Linear regression model for V29 as output
lm_v29_tt = lm(S11_V29~ length.of.patch + width.of.patch + height.of.patch + height.of.substrate + height.of.solder.resist.layer + radius.of.the.probe + c_pad + c_antipad + c_probe + dielectric.constant.of.substrate + dielectric.constant.of.solder.resist.layer, data = train_data2)
summary(lm_v29_tt)

# Making training and test data for V28 as output
set.seed(125) 
sample_index3 <- sample(1:nrow(hw1_input), 0.7 * nrow(hw1_input))
train_input3 <- hw1_input[sample_index3, ]
train_output3 <- magnitude_s11_v28[sample_index3]
test_input3 <- hw1_input[-sample_index3, ]
test_output3 <- magnitude_s11_v28[-sample_index3]
train_data3 <- data.frame(train_input3, train_output3)
test_data3 <- data.frame(test_input3, test_output3)
colnames(train_data3)[ncol(train_data3)] <- "S11_V28"

# Linear regression model for V28 as output
lm_v28_tt = lm(S11_V28~ length.of.patch + width.of.patch + height.of.patch + height.of.substrate + height.of.solder.resist.layer + radius.of.the.probe + c_pad + c_antipad + c_probe + dielectric.constant.of.substrate + dielectric.constant.of.solder.resist.layer, data = train_data3)
summary(lm_v28_tt)

# Making training and test data for average of V72 and V29 as output
combined_v72_v29_data <- data.frame(magnitude_s11_v72, magnitude_s11_v29)
magnitude_s11_v72_v29_avg <- rowMeans(combined_v72_v29_data, na.rm = TRUE)
set.seed(126) 
sample_index4 <- sample(1:nrow(hw1_input), 0.7 * nrow(hw1_input))
train_input4 <- hw1_input[sample_index4, ]
train_output4 <- magnitude_s11_v72_v29_avg[sample_index4]
test_input4 <- hw1_input[-sample_index4, ]
test_output4 <- magnitude_s11_v72_v29_avg[-sample_index4]
train_data4 <- data.frame(train_input4, train_output4)
test_data4 <- data.frame(test_input4, test_output4)
colnames(train_data4)[ncol(train_data4)] <- "S11_V72_V29"

# Linear regression model for average of V72 and V29 as output
lm_v72_v29_tt = lm(S11_V72_V29~ length.of.patch + width.of.patch + height.of.patch + height.of.substrate + height.of.solder.resist.layer + radius.of.the.probe + c_pad + c_antipad + c_probe + dielectric.constant.of.substrate + dielectric.constant.of.solder.resist.layer, data = train_data4)
summary(lm_v72_v29_tt)

# Making training and test data for average of V72, V29 and V28 as output
combined_v72_v29_v28_data <- data.frame(magnitude_s11_v72, magnitude_s11_v29, magnitude_s11_v28)
magnitude_s11_v72_v29_v28_avg <- rowMeans(combined_v72_v29_v28_data, na.rm = TRUE)
set.seed(127) 
sample_index5 <- sample(1:nrow(hw1_input), 0.7 * nrow(hw1_input))
train_input5 <- hw1_input[sample_index5, ]
train_output5 <- magnitude_s11_v72_v29_v28_avg[sample_index5]
test_input5 <- hw1_input[-sample_index5, ]
test_output5 <- magnitude_s11_v72_v29_v28_avg[-sample_index5]
train_data5 <- data.frame(train_input5, train_output5)
test_data5 <- data.frame(test_input5, test_output5)
colnames(train_data5)[ncol(train_data5)] <- "S11_V72_V29_V28"

# Linear regression model for average of V72, V28 and V29 as output
lm_v72_v29_v28_tt = lm(S11_V72_V29_V28~ length.of.patch + width.of.patch + height.of.patch + height.of.substrate + height.of.solder.resist.layer + radius.of.the.probe + c_pad + c_antipad + c_probe + dielectric.constant.of.substrate + dielectric.constant.of.solder.resist.layer, data = train_data5)
summary(lm_v72_v29_v28_tt)

# Linear regression model for average of V72 and V29 as output with removed variables
lm_v72_v29_tt = lm(S11_V72_V29~ width.of.patch + height.of.substrate + height.of.solder.resist.layer + radius.of.the.probe + c_antipad + c_probe + 
dielectric.constant.of.substrate + dielectric.constant.of.solder.resist.layer, data = train_data4)

# Linear regression model for average of V72 and V29 as output with further removed variables
lm_v72_v29_tt = lm(S11_V72_V29~ width.of.patch + height.of.substrate + height.of.solder.resist.layer + radius.of.the.probe + c_antipad + c_probe + dielectric.constant.of.substrate + dielectric.constant.of.solder.resist.layer, data = train_data4)
summary(lm_v72_v29_tt)

# Predicting S11 magnitude with test data
predictions <- predict(lm_v72_v29_tt, newdata = test_input4)
mse <- mean((test_output4 - predictions)^2)
print(mse)

# Predicting real and imaginary components with test data for V72 and V29
v72_real <- hw1_real[ ,72]
v72_img <- hw1_img[ ,72]
v29_real <- hw1_real[ ,29]
v29_img <- hw1_img[ ,29]
set.seed(126) 
test_output4_v72_real <- v72_real[-sample_index4]
test_output4_v29_real <- v29_real[-sample_index4]
test_output4_v72_img <- v72_img[-sample_index4]
test_output4_v29_img <- v29_img[-sample_index4]

predictions_v72_real <- predict(lm_v72_v29_tt, newdata = test_input4)
mse_v72_real <- mean((test_output4_v72_real - predictions_v72_real)^2)
print(mse_v72_real)

predictions_v72_img <- predict(lm_v72_v29_tt, newdata = test_input4)
mse_v72_img <- mean((test_output4_v72_img - predictions_v72_img)^2)
print(mse_v72_img)

predictions_v29_real <- predict(lm_v72_v29_tt, newdata = test_input4)
mse_v29_real <- mean((test_output4_v29_real - predictions_v29_real)^2)
print(mse_v29_real)

predictions_v29_img <- predict(lm_v72_v29_tt, newdata = test_input4)
mse_v29_img <- mean((test_output4_v29_img - predictions_v29_img)^2)
print(mse_v29_img)

# Building regression with real component dataset
real_v72 <- hw1_real[ ,72]
img_v72 <- hw1_img[ ,72]

set.seed(134) 
sample_index7 <- sample(1:nrow(hw1_input), 0.7 * nrow(hw1_input))
train_input7 <- hw1_input[sample_index7, ]
train_output7 <- real_v72[sample_index7]
test_input7 <- hw1_input[-sample_index7, ]
test_output7 <- real_v72[-sample_index7]
train_data7 <- data.frame(train_input7, train_output7)
test_data7 <- data.frame(test_input7, test_output7)

lm_v72_real = lm(train_output7~ length.of.patch + width.of.patch + height.of.patch + height.of.substrate + height.of.solder.resist.layer + radius.of.the.probe + c_pad + c_antipad + c_probe + dielectric.constant.of.substrate + dielectric.constant.of.solder.resist.layer, data = train_data7)
summary(lm_v72_real)

# Predicting the real component 
predictions_v72_real_2 <- predict(lm_v72_real, newdata = test_input7)
mse_v72_real_2 <- mean((test_output7 - predictions_v72_real_2)^2)
print(mse_v72_real_2)

# Building regression with img component dataset
set.seed(135) 
sample_index8 <- sample(1:nrow(hw1_input), 0.7 * nrow(hw1_input))
train_input8 <- hw1_input[sample_index8, ]
train_output8 <- img_v72[sample_index8]
test_input8 <- hw1_input[-sample_index8, ]
test_output8 <- img_v72[-sample_index8]
train_data8 <- data.frame(train_input8, train_output8)
test_data8 <- data.frame(test_input8, test_output8)

lm_v72_img = lm(train_output8~ length.of.patch + width.of.patch + height.of.patch + height.of.substrate + height.of.solder.resist.layer + radius.of.the.probe + c_pad + c_antipad + c_probe + dielectric.constant.of.substrate + dielectric.constant.of.solder.resist.layer, data = train_data8)
summary(lm_v72_img)

