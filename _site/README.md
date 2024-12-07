# Recipe Classifier
Project for EECS398 (Practical Data Science) at University of Michigan

By Zifei Bai

# Introduction

## General Introduction

Every country has tasty recipes. Users share their homemade recipes on Food.com to help others. But sometimes, the recipe has its unique name or directly translated pronunciation from other languages—for instance, kalbi (Korean beef rib). In this project, the primary goal is to build a recipe classifier that can predict the type of recipe based on its nutrition composition, which can help users from all over the world quickly identify the type of recipe they are not familiar with. This recipe classifier can classify recipes into three different categories—Starters, Main Courses and Desserts. 

First, I will clean the dataset, and do some exploratory data analysis to gain a basic understanding of this dataset, including data distribution and the nutrition composition of each type of recipe. 

Based on the understanding of exploratory data analysis, I will pick and design a baseline classification model and evaluate the performance of this model. After the baseline model, I will improve the model and measure the performance and improvement compared to the baseline model. 


## Introduction of Dataset

This dataset contains recipes and ratings from food.com. It was originally scraped and used by the authors of Generating Personalized Recipes from Historical User Preferences (UCSD). 

Since the original data is quite large, I will use the subset of the raw data used in the original report, containing only the recipes and reviews posted since 2008.

The original raw dataset has two DataFrames, `RECIPES` and `RATINGS`. `RECIPES` contains 83782 rows, corresponding to 83782 recipes, and 12 columns. `RATINGS` contains 731927  rows, as each recipe could have more than one review. However, I will only focus on a few of these columns for the sake of my analysis.

`RECIPES`

|Column                |Description|
|---                |---        |
|`'name'`                |Name of this recipe|
|`'id'`                |Recipe ID|
|`'nutrition'`                |Nutrition information in the form [calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV)]; PDV stands for “percentage of daily value”|
|`'minutes'`                |Minutes to prepare recipe|
|`'contributor_id'`                |User ID who submitted this recipe|
|`'submitted'`                |Date recipe was submitted|
|`'tags'`                |Food.com tags for recipe|
|`'n_steps'`                |Number of steps in recipe|
|`'steps'`                |Text for recipe steps, in order|
|`'description'`                |User-provided description|


`RATING`

|Column                |Description|
|---                |---        |
|`'recipe_id'`                |Recipe ID|
|`'rating'`                |Rating given|
|`'date'`	                 |Date of interaction|
|`'rating'`	                 |Rating given|
|`'review'`	                 |Review text|

# Data Cleaning and Exploratory Data Analysis

## Cleaning

1. I left merge `RECIPES` on `'id'` with `RATINGS` on `'recipe_id'`. Then, I dropped the `'recipe_id'` column, and grouped the columns with `'id'`. 

2. I found that there are several rows of `'rating'` that are 0, but when the user rates the recipe, the range is 1-5, so the data might be missed, or the rating part is optional, so the user didn’t give a rating for this review. I replaced all the 0s with np.nan because when computing the average rating of each recipe, np.nan doesn’t affect the average but 0s will lower the average. After filled the missing value, I computed the average rating of each recipe and then merged this series with the DataFrame. 

3. For now, I got the average rating for each recipe, so I dropped the duplicate rows base on `id`. The result DataFrame contains 83782 rows, which is same as the original `RECIPES`. 

4. The column `nutrition` contains a lot of information, I extract `calories` (#), `total fat` (PDV), `sugar` (PDV), `sodium` (PDV), `protein` (PDV), `saturated fat` (PDV), `carbohydrates` (PDV) and convert them into float and assign them to new columns. 

5. I dropped all the irrelevant columns, and the resulting DataFrame now contains 9 columns `’name’`, `avg_rating`, `’calories’`, `’sugar’`, `’protein’`, `’sodium’`, `’total_fat’`, `’saturated_fat’`, and `’carbohydrates’`.

6. Because the original dataset doesn’t contain the type of each recipe, I need to manually select some classic recipes for `’main_courses’`, `’desserts’`, and `’starters’`. My approach is to use regular expressions to filter keywords on the `’name’` column. For example, `’name’` contains ‘burger’ is `’main_courses`, `’name’` contains `’brownie’` is `’desserts’`, `’name’` contains `’appetizer’` is `’starters’`. I used a lot of keywords and tried to cover all the main classic types of the recipe as comprehensively as possible. After this process, the result DataFrame contains 15106 rows with the `’label’` `’main_courses’`, 11473 rows with the `’label’` `’desserts’`, and only 266 rows with the `’label’` `’starters’`. 


7. Because this dataset is unbalanced, I then duplicated the rows with the label `’starters’` 50 times, the result DataFrame now contains 13158 rows with `’label’` `’starters’`, 15106 rows with the `’label’` `’main_courses’`, 11473 rows with the `’label’` `’desserts’`, total 38913 rows and 10 columns, `’name’`, `avg_rating`, `’calories’`, `’sugar’`, `’protein’`, `’sodium’`, `’total_fat’`, `’saturated_fat’` `’carbohydrates’`, and `’label’`. 

8. Finally, I randomly selected 10000 rows for each type of recipe and the resulting DataFrame now contains 30000 rows and 10 columns. 

## Exploratory Data Analysis

### Univariate Analysis

In my exploratory data analysis, I first perform univariate analysis to examine the distribution of single variables.

I plotted the histograms for different nutrition types to explore the distribution of each nutrition type in all three types of recipes. 

<iframe
  src="assets/Protein-hist.html"
  width="800"
  height="450"
  frameborder="0"
></iframe>

<iframe
  src="assets/Sugar-hist.html"
  width="800"
  height="450"
  frameborder="0"
></iframe>

<iframe
  src="assets/Carbo-hist.html"
  width="800"
  height="450"
  frameborder="0"
></iframe>

### Bivariate Analysis

I would like to know the distribution of `’protein’`, `’sugar’`, and `’carbohydrate’` content for `’desserts’`, `’main_courses’` and `’starters’`. The graphs below show the amount of different nutrients in the three recipes. 
 

<iframe
  src="assets/bivariate-protein.html"
  width="800"
  height="450"
  frameborder="0"
></iframe>

<iframe
  src="assets/bivariate-sugar.html"
  width="800"
  height="450"
  frameborder="0"
></iframe>

<iframe
  src="assets/bivariate-carbohydrates.html"
  width="800"
  height="450"
  frameborder="0"
></iframe>

It shows that `’desserts’` tend to have more `’sugar’`, `’main_courses’` tend to have more `’protein’`, and `’starters’` contain less of these three nutrients. 

I found that the distribution of these three nutrients is well characterized and can be used in subsequent classification models. I then plotted the 3D scattered plot to show the distribution of these three recipe types, the x, y, and z axes are `’sugar’`, `’protein’`, and `’carbohydrates’`. 

<iframe
  src="assets/bivariate-3d-scatter.html"
  width="1000"
  height="450"
  frameborder="0"
></iframe>

### Grouping and Aggregates

I generated a pivot table to see for each type of recipe, the average `avg_rating`, `calories`, `carbohydrates`, `protein`, `sodium`, and `sugar`. 

|label               |avg_rating|   calories  |carbohydrates|protein|sodium|sugar|
|---                 |---       |---          |---          |---    |---   |---  |
|`'desserts'`        |4.16|         357.63    |14.30        |12.97  |12.49 |98.06|
|`'mian_courses'`    |4.40|         459.14    |10.76        |55.82  |34.57 |28.25|
|`'starters'`	     |4.51|         279.51    |6.71         |20.53  |21.63 |22.68|



# Baseline Model

First, I chose  Multinomial Logistic regression model to do the classification. I trained this model using `’protein’`(quantitative), `’sugar’`(quantitative), and `’carbohydrates’`(quantitative). 

I split the dataset into the training set and test set, 80% of the data is in the training set and 20% of the data is in the test set.


I made a pipeline, first normalized these three features, and then fit multinomial logistic regression model. 

The macro average of accuracy, precision, recall, and F1-score are:



|---                |---        |
|Accuracy                |0.7373|
|Precision                |0.75 |
|Recall                |0.74 |
|F1-score              |0.74  |

<br/>

The resulting confusion matrix for the multinomial logistic regression model is shown below. 

<iframe
  src="assets/base-cm.html"
  width="850"
  height="450"
  frameborder="0"
></iframe>

# Final Model

I found that the performance of the Multinomial Logistic Regression Model in this dataset is not very well. I hypothesized that the reason the logistic regression model did not perform well on this dataset is that the decision boundary for the logistic regression model must be linear, but this dataset, because the three recipes would overlap in some feature space and couldn't be separated linearly, therefore a linear decision boundary that did not do a good job of dividing the three types of recipes into three regions.

After examining the characteristics of decision boundaries for Logistic Regression, K-NN, and RandomForst, and Exploring the distribution of the data, I finally chose to use the RandomForest Classification Model. This model can divided into different ‘rectangular-like’ modules, in my model, the decision boundaries are some cuboids as I have three features. 

I made a new pipeline, using RandomForest Classifier, and then using Cross-Validation to determine four hyperparameters by finding the highest macro F1-score: `max_depth`, `min_samples_split`, `criterion`, and `n_estimators`. 


The result of Cross-Validation is:
- `max_depth`: 18
- `min_samples_split`: 10
- `criterion`: 'entropy'
- `n_estimators`: 60


The macro average of accuracy, precision, recall, and F1-score are:



|---                |---        |
|Accuracy                |0.9168|
|Precision                |0.92 |
|Recall                |0.92  |
|F1-score              |0.92   |



The resulting confusion matrix for the multinomial logistic regression model is shown below.
<br/>

<iframe
  src="assets/final-cm.html"
  width="850"
  height="450"
  frameborder="0"
></iframe>

Compared to my baseline model, the overall performance improved from 75% to over 90%, which is a huge improvement. 


RandomForest can divide the feature space into different 'rectangular-like' regions, so it is more flexible than Logistic Regression. I also used Cross-Validation to find the best hyperparameters of the RandomForest Classification Model, which also allowed my final model to perform better than the baseline model's prediction. 
