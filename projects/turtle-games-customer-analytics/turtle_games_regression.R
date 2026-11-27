# Turtle Games: exploratory analysis and multiple linear regression in R
# Author: Lan Ngoc Mau
# Tools: R (tidyverse, ggplot2, plotly, skimr, DataExplorer, psych, moments, car)
#
# Part 5  Exploratory analysis: how do loyalty points relate to income,
#         spending score, age, gender and education?
# Part 6  Multiple linear regression to predict loyalty points, with
#         normality checks, multicollinearity (VIF) and model diagnostics.
#
# Packages needed (install once with install.packages()):
#   tidyverse, skimr, scales, ggplot2, DataExplorer, plotly, moments, psych, car
#
# Data file expected in the working directory (not included in this repo):
#   turtle_reviews.csv

################################################################################
# Part 5: Exploratory data analysis
################################################################################

##  Import all the required libraries for the analysis and view the data. 
library(tidyverse)
library(skimr)
library(scales)
library(ggplot2)
library(DataExplorer)
## Load and explore the data.
review <- read.csv('turtle_reviews.csv', header = TRUE)
## View the first rows of the data.
head(review)
## Explore how loyalty points relate to income, spending score and demographics
## with scatterplots, histograms and boxplots.
# View the columns of dataframe
colnames(review)
# Change columns' name
review <- review %>% rename(annual_income = annual_income..k..,
                            customer_score = customer_score..1.100.)
# View the columns name after the change
colnames(review)
# Understanding the range of numeric columns value.
summary(review)
## Create scatterplot to understand relationship between loyalty_points and annual income
plot1 <- ggplot(data = review,
       mapping = aes(x = loyalty_points, y = annual_income)) +
  geom_point(alpha = 0.5, size = 1.5, color = 'orange') +
  geom_smooth(method = 'lm', se = FALSE, size = 1.5) + 
  scale_x_continuous(breaks = seq(0, 8000, 2000)) +
  scale_y_continuous(breaks = seq(0, 120, 20),
                     labels = label_number(suffix = "k£")) +
  labs(title = "Relationship between Annual Income and Loyalty Points",
       subtitle = "Customers with more loyalty points tend to have bigger annual income" ,
       #  Add labels to labs function.
       caption = "Source: Turtle Games Sales Data",
       x = "Loyalty Points",
       y = "Annual Income of Customers (k£)") +
  
  # Add a theme layer. 
  theme_bw() + 
  theme(plot.title = element_text(face = "bold"),
        plot.subtitle = element_text(face = "italic"),
        plot.caption = element_text (face= 'italic'),
        axis.title.x = element_text( face= 'bold'),
        axis.title.y = element_text( face= 'bold'))
  # Adding interactive visualisations 
library(plotly)
ggplotly(plot1)

## Create scatterplot to understand relationship between loyalty_points and annual income, divided by gender
plot2 <- ggplot(data = review,
       mapping = aes(x = loyalty_points, y = annual_income, color = gender)) +
  geom_point(alpha = 0.5, size = 1.5) +
  geom_smooth(method = 'lm', se = FALSE, size = 1.5) + 
  scale_x_continuous(breaks = seq(0, 8000, 2000)) +
  scale_y_continuous(breaks = seq(0, 120, 20),
                     labels = label_number(suffix = "k£")) +
  scale_color_manual(values = c( Male = '#00BFC4', Female = '#F8766D'), labels = c('Male', 'Female')) +
  facet_wrap(~gender)+
  labs(title = "Relationship between Annual Income and Loyalty Points based on Gender",
       subtitle = "Higher Loyalty Points are associated with greater Annual Income - a consistent pattern for both genders",
       #  Add labels to labs function.
       caption = "Source: Turtle Games Sales Data",
       x = "Loyalty Points",
       y = "Annual Income of Customers (k£)",
       color = "Gender") +
  
  # Add a theme layer. 
  theme_bw() + 
  theme(plot.title = element_text(face = "bold"),
        plot.subtitle = element_text(face = "italic"),
        plot.caption = element_text (face= 'italic'),
        axis.title.x = element_text( face= 'bold'),
        axis.title.y = element_text( face= 'bold'),
        strip.text = element_text(face = "bold"))

## Adding visualisations
ggplotly(plot2)

## Create scatterplot to understand relationship between loyalty_points and annual income, divided by education
unique(review$education)
levels(factor(review$education))
plot3 <- ggplot(data = review,
       mapping = aes(x = loyalty_points, y = annual_income, color = education)) +
  geom_point(alpha = 0.5, size = 1.5) +
  geom_smooth(method = 'lm', se = FALSE, size = 1.5) + 
  scale_x_continuous(breaks = seq(0, 8000, 2000)) +
  scale_y_continuous(breaks = seq(0, 120, 20),
                     labels = label_number(suffix = "k£")) +
  scale_color_manual(values = c( graduate = '#00BFC4', postgraduate = '#F8766D', diploma = '#C77CFF', Basic = '#7CAE00', PhD = '#00B0F6'),
                     labels = c(graduate = 'Graduate', 
                                PhD = 'PhD', 
                                diploma = 'Diploma', 
                                postgraduate = 'Postgraduate', 
                                Basic = 'Basic'))  +
  facet_wrap(~education,labeller = as_labeller(c(graduate = "Graduate", PhD= "PhD",
                                                 diploma = "Diploma", postgraduate = "Postgraduate", Basic = "Basic"))) +
  labs(title = "Relationship between Annual Income and Loyalty Points based on Education Level",
       subtitle = "Loyalty Points rise with Annual Income across all education levels with the majority of customers are university-educated",
       #  Add labels to labs function.
       caption = "Source: Turtle Games Sales Data",
       x = "Loyalty Points",
       y = "Annual Income of Customers (k£)",
       color = "Education Level") +
  
  # Add a theme layer. 
  theme_bw() + 
  theme(plot.title = element_text(face = "bold"),
        plot.subtitle = element_text(face = "italic"),
        plot.caption = element_text (face= 'italic'),
        axis.title.x = element_text( face= 'bold'),
        axis.title.y = element_text( face= 'bold'),
        strip.text = element_text(face = "bold"))

# Adding interactive visualisations
ggplotly(plot3)
  
## Create scatterplot to understand relationship between loyalty_points and customer score
plot4 <- ggplot(data = review,
       mapping = aes(x = loyalty_points, y = customer_score)) +
  geom_point(alpha = 0.5, size = 1.5, color = 'darkgreen') +
  geom_smooth(method = 'lm', se = FALSE, size = 1.5) + 
  scale_x_continuous(breaks = seq(0, 8000, 2000)) +
  scale_y_continuous(breaks = seq(0, 120, 20)) +
  labs(title = "Relationship between Customer Score and Loyalty Points",
       subtitle = "Customer Score and Loyalty Points show a strong positive correlation, however lower-engagement customers show inconsistent scores" ,
       #  Add labels to labs function.
       caption = "Source: Turtle Games Sales Data",
       x = "Loyalty Points",
       y = "Customer Score") +

  # Add a theme layer. 
  theme_bw() + 
  theme(plot.title = element_text(face = "bold"),
        plot.subtitle = element_text(face = "italic"),
        plot.caption = element_text (face= 'italic'),
        axis.title.x = element_text( face= 'bold'),
        axis.title.y = element_text( face= 'bold'))
# Adding interactive visualisations
ggplotly(plot4)

## Create scatterplot to understand relationship between loyalty_points and customer score, divided by gender
plot5 <- ggplot(data = review,
       mapping = aes(x = loyalty_points, y = customer_score, color = gender)) +
  geom_point(alpha = 0.5, size = 1.5) +
  geom_smooth(method = 'lm', se = FALSE, size = 1.5) + 
  scale_x_continuous(breaks = seq(0, 8000, 2000)) +
  scale_y_continuous(breaks = seq(0, 100, 20))+
  scale_color_manual(values = c( Male = '#5C6BC0', Female = '#FF8A65'), labels = c('Male', 'Female')) +
  facet_wrap(~gender)+
  labs(title = "Relationship between Customer Score and Loyalty Points based on Gender",
       subtitle = "Loyalty Points rise with Customer Score similarly for both genders",
       #  Add labels to labs function.
       caption = "Source: Turtle Games Sales Data",
       x = "Loyalty Points",
       y = "Customer Score",
       color = "Gender") +
  
  # Add a theme layer. 
  theme_bw() + 
  theme(plot.title = element_text(face = "bold"),
        plot.subtitle = element_text(face = "italic"),
        plot.caption = element_text (face= 'italic'),
        axis.title.x = element_text( face= 'bold'),
        axis.title.y = element_text( face= 'bold'),
        strip.text = element_text(face = "bold"))
## Adding interactive visualisations
ggplotly(plot5)

## Create histogram to understand loyalty points by gender
plot6 <- ggplot(data = review, mapping = aes(x = loyalty_points, fill = gender)) +
  geom_histogram(binwidth = 500, alpha = 0.6, position = "identity") +
  scale_fill_manual(values = c(Male = '#5C6BC0', Female = '#FF8A65')) +
  facet_wrap(~gender) +
  labs(title = "Distribution of Loyalty Points by Gender",
       subtitle = "Loyalty points are heavily concentrated at lower values for both genders",
       x = "Loyalty Points", y = "Number of Customers", fill = "Gender",
       caption = "Source: Turtle Games Sales Data") +
  theme_bw() +
  theme(plot.title = element_text(face = "bold"),
        plot.subtitle = element_text(face = "italic"),
        strip.text = element_text(face = "bold"),
        plot.caption = element_text (face= 'italic'),
        axis.title.x = element_text( face= 'bold'),
        axis.title.y = element_text( face= 'bold'))

# Adding interactive visualisations
ggplotly(plot6)

## Create histogram to understand loyalty points by education
plot7 <- ggplot(data = review, mapping = aes(x = loyalty_points, fill = education)) +
  geom_histogram(binwidth = 500, alpha = 0.6, position = "identity") +
  scale_fill_manual(values = c(graduate = '#00BFC4', PhD = '#F8766D', diploma = '#C77CFF', 
                               postgraduate = '#7CAE00', Basic = '#00B0F6'),
                    labels = c(graduate = 'Graduate', PhD = 'PhD', diploma = 'Diploma',
                               postgraduate = 'Postgraduate', Basic = 'Basic')) +
  facet_wrap(~education, labeller = as_labeller(c(graduate = "Graduate", PhD = "PhD",
                                                  diploma = "Diploma", postgraduate = "Postgraduate", 
                                                  Basic = "Basic"))) +
  labs(title = "Distribution of Loyalty Points by Education Level",
       subtitle = "Graduate customers make up the majority number of overall customers",
       x = "Loyalty Points", y = "Number of Customers", fill = "Education Level",
       caption = "Source: Turtle Games Sales Data") +
  theme_bw() +
  theme(plot.title = element_text(face = "bold"),
        plot.subtitle = element_text(face = "italic"),
        strip.text = element_text(face = "bold"),
        plot.caption = element_text(face = 'italic'),
        axis.title.x = element_text(face = 'bold'),
        axis.title.y = element_text(face = 'bold'))

# Adding interactive visualistions
ggplotly(plot7)

## Understanding the age range in customer base and their education level:
plot8 <- ggplot(data = review, mapping = aes(x = education, y = age, fill = education)) +
  geom_boxplot(alpha = 0.6) +
  scale_fill_manual(values = c(graduate = '#00BFC4', PhD = '#F8766D', diploma = '#C77CFF', 
                               postgraduate = '#7CAE00', Basic = '#00B0F6'),
                    labels = c(graduate = 'Graduate', PhD = 'PhD', diploma = 'Diploma',
                               postgraduate = 'Postgraduate', Basic = 'Basic')) +
  scale_x_discrete(labels = c(Basic = "Basic", diploma = "Diploma", graduate = "Graduate", 
                              PhD = "PhD", postgraduate = "Postgraduate"))+ 
  scale_y_continuous(breaks = seq(0, 70, 10)) + 
  labs(title = "Distribution of Age by Education Level",
       subtitle = "Customers of Turtle Games are mostly above 30s",
       x = "Education Level", y = "Age", fill = "Education Level",
       caption = "Source: Turtle Games Sales Data") +
  theme_bw() +
  theme(plot.title = element_text(face = "bold"),
        plot.subtitle = element_text(face = "italic"),
        strip.text = element_text(face = "bold"),
        plot.caption = element_text(face = 'italic'),
        axis.title.x = element_text(face = 'bold'),
        axis.title.y = element_text(face = 'bold'))

# Adding interactive visualisations
ggplotly(plot8)

## Understanding the relationship between income and gender:
plot9 <- ggplot(data = review, mapping = aes(x = gender, y = annual_income, fill = gender)) +
  geom_boxplot(alpha = 0.6) +
  scale_y_continuous(labels = label_number( suffix = "k£")) +
  scale_fill_manual(values = c(Male = '#5C6BC0', Female = '#FF8A65')) +
  labs(title = "Distribution of Annual Income by Gender",
       subtitle = "Gender does not reflect strongly the differentiator of annual income",
       x = "Gender", y = "Annual Income (k£)", fill = "Gender",
       caption = "Source: Turtle Games Sales Data") +
  theme_bw() +
  theme(plot.title = element_text(face = "bold"),
        plot.subtitle = element_text(face = "italic"),
        strip.text = element_text(face = "bold"),
        plot.caption = element_text(face = 'italic'),
        axis.title.x = element_text(face = 'bold'),
        axis.title.y = element_text(face = 'bold'))

# Adding interactive visualisations
ggplotly(plot9)

###############################################################################

# 1. Demographic Patterns:

# - Gender shows minimal differentiation: by plotting and understanding the relationship of 
# annual income/customer score to loyalty points
# it is suggested that gender barely has any influence on annual income/customer score/loyalty points
# -> This is also the same conclusion in previous findings about what factors might need to consider
# in understanding how customer accumulate loyalty points.
# - Majority of customers are above 30s, rarely under 30s, with graduate customers made up the largest segment.
# -> Significant imbalance between customers with basic educational level and those with higher degrees

# 2. Loyalty points as the Central Driver
# - Loyalty points correlate strongly and positively with both annual income and customer score
# regardless of gender or education level.
# - At low loyalty point level  (0-2000), customer score and income vary widely
# some low-engagement customers score as low as 0-20 while others score 80-100 
# a noticeable amount of customers with higher income (more than 60k£)
# indeed have quite low loyalty points (0-2000) -> this need to be studied further
# As loyalty points increase beyond 2000, the relationship become more consistent

# 3. Findings and Insights summary:
# 3.1 Loyalty points are the strongest and most consistent correlate of 
# both income and customer score across every segment tested
# 3.2 Gender is not a meaningful differentiator of income, loyalty engagement, or score
# 3.3 Most customers have at least university-degree
# 3.4 A noticeable amount of customers, across all segments, sit in the lower loyalty point range (0-2,000 points) 

# 4. Recommendations
# 4.1 Using the reviews and feedback received from customers, understanding about
# what needs to be improved and suggested to move customers from low-engagement tier (0-2000)
# to higher tiers

# 4.2 Should be targeting more between customers score and annual income relationship
# to loyalty points given their consistent correlation

###############################################################################

################################################################################
# Part 6: Multiple linear regression to predict loyalty points
# Questions:
#  - Can a relatively simple MLR model predict loyalty points from the existing features?
#  - How good is the fit, and can the results be trusted?
#  - Where should the business focus its marketing, and how could the loyalty programme improve?
################################################################################

# Load the dataframe
head(review)

## Continue with the data frame prepared in Part 5.
dim(review)

## Descriptive statistics: how are loyalty points distributed?
summary(review)
dev.off()
# Check the Normal Distribution of loyalty points
# Specify the qnorm function
# Draw a qqplot using the loyalty_points
qqnorm(review$loyalty_points,
       col= 'darkgreen',
       main = "Normal Distribution of Loyalty Points",
       xlab= expression(italic("Theoretical Quantiles")),
       ylab= expression(italic('Customer Loyalty Points')))

# Add a reference line to the qqplot.
qqline(review$loyalty_points,
       col='red',
       lwd=2) 

# Run a Shapiro-Wilk test
shapiro.test(review$loyalty_points)

library(moments)

# Specify the skewness and kurtosis functions.
skewness(review$loyalty_points) 
kurtosis(review$loyalty_points)

## Distribution and patterns:
## Shapiro-Wilk: W = 0.843, p < 2.2e-16 — strongly rejects the null hypothesis of normality (p far below 0.05). 
## loyalty_points is not normally distributed.
## Skewness = 1.46 — positive/right skew (values > 1 is considered highly skewed).
## -> Suggest the long right tail in the Q-Q plot

# Check the correlation of annual income and loyalty points
cor(review$annual_income, review$loyalty_points)

# Check the correlation of customer score and loyalty points
cor(review$customer_score, review$loyalty_points)

# Check the correlation of age vs loyalty points
cor(review$age, review$loyalty_points)

## Feature selection, concerns and corrective actions:
## The correlation score between annual income and loyalty points is 0.61 -> good feature of linear regression model
## The correlation score between customer score and loyalty points is 0.67 -> good feature of linear regression model
## The correlation score between age and loyalty points is -0.04 -> no correlation, this feature should not be included in the linear regression model
## Above findings also suggest gender is not a meaningful feature to be used for model.
## Concerns and corrective actions:
### Loyalty points is not normally distributed -> worth investigating
### Check for possible multicollinearity in case exist between customer score and annual income

## Multiple linear regression models

# Visualise the correlation

# Import the psych package.
library(psych)

# Create a linear regression model between annual income and loyalty points
# specify the lm function and the variables
model1 = lm(loyalty_points~annual_income, data=review)

# Print the summary statistics.
summary(model1)

# Create a linear regression model between customer score and loyalty points
# specify the lm function and the variables
model2 = lm(loyalty_points~customer_score, data=review)

# Print the summary statistics.
summary(model2)

# Create a linear regression model that combined of both customer score and annual income
# in predicting loyalty points
model3 = lm(loyalty_points~customer_score + annual_income, data=review)

# Print the summary statistics.
summary(model3)

# Try adding education
model4 = lm(loyalty_points~customer_score + annual_income + education, data=review)

# Print the summary statistics.
summary(model4)

# Try adding gender and age variables
model5 = lm(loyalty_points~customer_score + annual_income + education+ gender + age, data=review)

# Print the summary statistics.
summary(model5)

## Goodness of fit:
## Result of R-squared in different models:
## + annual_income only: model1 - 0.3795
## + customer_score only: model2 - 0.452
## + annual_income + customer_score: model3 -  0.8269
## + annual_income + customer_score + education: 0.8312
## + annual_income + customer_score + education + gender + age: 0.8439

## By building multiple linear regression models, it is suggested that
## the variance in customer loyalty points can be explained around 
## 83% by using annual income and customer score together
## Decided to use model3 as the final model as model3 could capture the majority of 
## explainable variance (adding other variables such as age, gender or education
## did not generate much difference in the prediction power)

# Check the summary of model3
summary(model3)

dev.off()
# Check assumptions
par(mfrow= c(2,2))
plot(model3)

# Check multicollinearity
library(car)
vif(model3)

## VIF score of customer_score and annual_income at 1 suggests that
## these 2 variables are not correlated to each other.

# Create a dataframe with only the customer score and annual income  to test the model
review_test <- review %>% select(customer_score, annual_income)

# View the dataframe
head(review_test)

# Explore the dataset
dim(review_test)
colnames(review_test)

# Create a new object and specify the predict function.
predictTest = predict(model3, newdata=review_test,
                      interval='confidence')

# Print the object
predictTest

# Evaluate the prediction accuracy
comparison <- cbind(review_test, predictTest)
comparison$actual_loyalty_points <- review$loyalty_points
comparison$difference <- comparison$actual_loyalty_points - comparison$fit

comparison

# Common accuracy metrics
mean(abs(comparison$difference))  # MAE
sqrt(mean(comparison$difference^2))# RMSE

# Getting mean and range of original loyalty points
mean(review$loyalty_points)
range(review$loyalty_points)

# Run model diagnostics
par(mfrow = c(2, 2))
plot(model3)

##  Create a visual demonstration of the model

ggplot(comparison, aes(x = actual_loyalty_points, y = fit)) +
  geom_point(alpha = 0.4, color = 'darkgreen') +
  geom_abline(intercept = 0, slope = 1, color = 'coral') +
  labs(title = "Predicted vs Actual Loyalty Points", 
       subtitle = " Model predicted Loyalty Points using Annual Income and Customer Score",
       caption = " Variance in Loyalty Points can be explained around 83% by using Annual Income and Customer Score ",
       x = "Actual Loyalty Points",
       y = "Predicted Loyalty Points") +
      
  # Add a theme layer. 
       theme_bw() + 
         theme(plot.title = element_text(face = "bold"),
               plot.subtitle = element_text(face = "italic"),
               plot.caption = element_text (face= 'italic'),
               axis.title.x = element_text( face= 'bold'),
               axis.title.y = element_text( face= 'bold'))

## Usefulness of the model and possible improvements:

## The mean loyalty points: 1578
## The range of loyalty points: 25 to 6847
## Mean absolute error: 414.83/ 1578.03 = 26.3% average error
## Root mean square error (showing how far off on average the model differs from actual values): 
## 533.74/1578.03 = 33.8% average error

## This could be interpreted as:
## - Loyalty range is huge (from 25 to 6847) 
## - The model managed to predict the overall trend well with R-squared = 83%
## - Model performs reasonably well for a large amount of customers
## for actual loyalty points between 1000 and 3000 (green points cluster around reference line)
## - The result also reveal 2 main limitations of the model:
## + low end where actual loyalty points near 0: model predicted several negative values
## + high end where actual loyalty points around 4000 upward: predicted values fall below the reference line
## which mean the predicted values failed to capture the actual loyalty points 
## In short, the model overestimates customers at the low end (even predicting negative points)
## and underestimates the highest-value customers, missing how many points they accumulate.
## -> This could be explained as why this model suggested 26-33% average error in individual loyalty points prediction as above

## Business Interpretations:
## 1. Annual income and customer score altogether could be used to explain 83% variation in loyalty programs
### -> 2 main drivers for loyalty points accumulation.
## 2. The model works well in predicting typical/mid-tier customers.
### -> the loyalty program is behaving as expected for a large portion of customer base.
## 3. The model fails to predict low-score customers (negative loyalty points)
## 4. High-value customers (4000+ points) with high loyalty points are underestimated by the model
### -> Actual loyalty points were actually beyond the predicted points by the model,
## which means something beyond income and customer also contribute to this.

## Business Recommendations:
### 1. The model can be used for mid-tier customer forecasting/ general narratives
### for low-engagement and high-value customers - the model might give misleading results.
### 2. Expand further variables beyond current dataset for example purchase frequency/preference/
### or any gift program that might help the model generate higher prediction power.
### 3. Revisit and validate the model to see if the model might perform better once the dataset
### is getting bigger.

###############################################################################
###############################################################################
