# Phase-3-Final-Project  
ByL Kristen Davis   

<p align="center">
  <img width="600" height="250" src="/Photos/photo2.jpeg">
</p>
 
 [](/Photos/photo2.jpeg)  

# Overview 
This project was created by Kristen Davis at Flatiron School for the Phase 3 Final Project - November 2020. This project was completed using predictive machine learning techniques as part of DataDriven's competition "Pump It Up: Data Mining the Water Table". The project goal was to use EDA to gain a firm understanding of a large dataset and then apply that to machine learning modeling.

  Skills Demonstrated within this project include: 
  * Proficiency in the Pandas, Plotly, Seaborn, Matplotlib, Numpy and sklearn libraries 
  * Advanced understanding of EDA & feature engineering techniques 
  * Machine Learning Techniques such as Decision Trees, Random Forest, GridSearchCV, Cost Function, RSME/ RME,  ROC curves, Feature Importance and Confusion      Matrices
  * Ability to present non technical data insights & data analysis 

# Scenario  
This project aimed to use data collected from from Taarifa and the Tanzanian Ministry of Water to predict the status of water pumps through out Tanzania. For each water pump the data set included a large number of information including- 

* Geographical data such as: latitude, logitude, region and subvillage 
* Water data such as: quality, source type, and quantity 
* Engineering data such as: total static head, pump type, and year constructed 
* Admministrative data such as: installer, funder, who manages the up, and if the pump has a pay structure 

The stated goal is to classify whether a water pump is: funtional, in need or repair, or broken based on the features provided. 

# The International Water Crisis  
Water scarcity is not a new issue, but it is one the world has failed to adequately address. According to the World Wildlife Foundation, "some 1.1 billion people worldwide lack access to water, and a total of 2.7 billion find water scarce for at least one month of the year. Inadequate sanitation is also a problem for 2.4 billion people—they are exposed to diseases, such as cholera and typhoid fever, and other water-borne illnesses." Tanzania, where nearly 50 percent of all water wells are in disrepair, suffers greatly from this crisis.   

Yet, with crisis comes opportunity -- opportunity to help, to make something better. Fixing the wells is a good first step, but it's nonetheless reactive. A more proactive approach --  Accurately predicting water pump functionality, therefore providing more consistent access to clean water and sanitary services -- would dramatically improve the lives of many, especially women and girls.

# EDA Analysis  
Below is a brief summary of the extensive EDA preformed on this dataset. In conjuction with modeling, these questions helped be understand the 'picture' of functioning and non functioning water wells in Tanzania.   

* 11% of all wells were funded by the government of Tanzania 

* If you consider the functional needs repair as functional all basins have more functioning wells than no functioning.

* 87% of wells had a public meeting

* 59% of the wells are managed by VWC

* 67% of the wells have a permit

* 55% of the wells use gravity as their extraction type

* 77% of wells have ground water as their water source 

* 67% of wells have a communal standpipe 

* The majority of wells that are functioning well with sufficient water are funded by the Government of Tanzania but if you look at all the wells the Government of Tanzania is making they are making more bad wells than good wells. Not good.

* For water wells producing water not classified as 'good' the biggest issue is the saltiness 

* The majority of unknown quality pumps are from the internal basin 

* It is almost a 50/50 split if a communal standpipe is working or not. If it is never pay much higher precentages of funtioning pipes in pay per bucket and pay monthly models (16%/ 16%)

* 6052 of 22824 total water pumps that are not functioning also are 'dry' 
* 37 of 4317 total water pumps that are in need of repair also are 'dry' 

# Modeling  
Modeling this data provided an oppurtunity to explore multiple modeling techniques: KNN, Logistic Regression, Random Forests and other ensemble methods. Through out modeling my goal was to identify the model, feature and parameters that would best predict the status of well in Tanzania. Within each model I took approach, enumerating on a variety of models and data processing techniques. Below are the testing accuracy scores for each of my models, as well as a detailed summary of my final model. 
 
 * Single Tree Vanilla Processing - Testing Accuracy for Decision Tree Classifier: 70.88%  
 
 * Bagged Decision Tree Vanilla Processing - Testing Accuracy for Decision Tree Classifier: 71.27% 
 
 * Random Forest Vanilla Processing - Testing Accuracy for Random Forest Classifier: 55.47% 
 
 * Random forest Grid Search: Testing Accuracy for Random Forest Classifier: 74.14%  
 
                      Optimal Parameters: {'criterion': 'gini', 'max_depth': None, 'min_samples_leaf': 3, 'min_samples_split': 5, 'n_estimators': 100} 
                      
 * KNN "Funtional Needs Repair" Binned to Functional - Testing Accuracy for KNN Classifier 54.77% 
 
 * Gausian Bayes "Funtional Needs Repair" Binned to Functional - Testing Accuracy for Gausian Classifier 72.93%  
 
 * Logistic Regression "Funtional Needs Repair" Binned to Functional - Testing Accuracy for Logistic Regression was 78% 
 
 In addition to each of these models I also completed several models using pipelines built in sklearn. These produced similarly accurate models (low 70s) but had a much more condenced workflow.

## Final Model  

## Summary  

## Future Work:  
* Use classes in workflow  
* Use sklearn pipelines throught out modeling  
* Use binary classification models with "Functional Needs Repair" binned with NonFunctioing  
* Stack multiple models 

# Resources: 
* Flatiron School Curriculum 
* DataDriven.org 
* Unsplash 

# [Blog Post](https://medium.com/@kristendavis27/graphing-feature-importance-with-scatter-polar-plots-72e9d0cb1d9c) 


