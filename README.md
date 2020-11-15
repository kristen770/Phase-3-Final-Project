# Phase-3-Final-Project  
ByL Kristen Davis 

# Overview 
This project was created by Kristen Davis at Flatiron School for the Phase 3 Final Project - November 2020. This project was completed using predictive machine learning techniques as part of DataDriven's competition "Pump It Up: Data Mining the Water Table". The project goal was to use EDA to gain a firm understanding of a large dataset and then apply that to machine learning modeling.

  Skills Demonstrated within this project include: 
  * Proficiency in the Pandas, Plotly, Seaborn, Matplotlib, Numpy and sklearn libraries 
  * Advanced understanding of EDA & feature engineering techniques 
  * Machine Learning Techniques such as Decision Trees, Random Forest, GridSearchCV, Cost Function, RSME/ RME,  ROC curves, Feature Importance and Confusion      Matrices
  * Ability to present non technical data insights & data analysis 

# Scenario  
<p align="center">
  <img width="600" height="250" src="/Photos/photo2.jpeg">
</p>
 
 [](/Photos/photo2.jpeg)  

This project aimed to use data collected from from Taarifa and the Tanzanian Ministry of Water to predict the status of water pumps through out Tanzania. For each water pump the data set included a large number of information including- 

* Geographical data such as: latitude, logitude, region and subvillage 
* Water data such as: quality, source type, and quantity 
* Engineering data such as: total static head, pump type, and year constructed 
* Admministrative data such as: installer, funder, who manages the up, and if the pump has a pay structure 

The stated goal is to classify whether a water pump is: funtional, in need or repair, or broken based on the features provided. 

# The International Water Crisis  
<p align="center">
  <img width="600" height="250" src="/Photos/image1.jpeg">
</p>
 
 [](/Photos/image1.jpeg)  

Water scarcity is not a new issue, but it is one the world has failed to adequately address. According to the World Wildlife Foundation, "some 1.1 billion people worldwide lack access to water, and a total of 2.7 billion find water scarce for at least one month of the year. Inadequate sanitation is also a problem for 2.4 billion people—they are exposed to diseases, such as cholera and typhoid fever, and other water-borne illnesses." Tanzania, where nearly 50 percent of all water wells are in disrepair, suffers greatly from this crisis.   

<p align="center">
  <img width="600" height="400" src="/Photos/Screen%20Shot%202020-11-12%20at%206.27.46%20PM.png">
</p>
 
 [](/Photos/Screen%20Shot%202020-11-12%20at%206.27.46%20PM.png)

Yet, with crisis comes opportunity -- opportunity to help, to make something better. Fixing the wells is a good first step, but it's nonetheless reactive. A more proactive approach --  Accurately predicting water pump functionality, therefore providing more consistent access to clean water and sanitary services -- would dramatically improve the lives of many, especially women and girls.

# EDA Analysis  
<p align="center">
  <img width="600" height="250" src="/Photos/photo5.jpeg">
</p>
 
 [](/Photos/photo5.jpeg)   
 
 <p align="center">
  <img width="600" height="250" src="/Photos/newplot.png">
</p>
 
 [](/Photos/newplot.png)   

# Modeling  
<p align="center">
  <img width="600" height="250" src="/Photos/photo4.jpeg">
</p>
 
 [](/Photos/photo4.jpeg) 
 
 Modeling this data provided an oppurtunity to explore multiple modeling techniques: KNN, Logistic Regression, Random Forests and other ensemble methods. Through out modeling my goal was to identify the model, feature and parameters that would best predict the status of well in Tanzania. Within each model I took approach, enumerating on a variety of models and data processing techniques. Below are the testing accuracy scores for each of my models, as well as a detailed summary of my final model. 
 
 * Single Tree - Testing Accuracy for Decision Tree Classifier: 70.88%  

# Final Model 
<p align="center">
  <img width="600" height="250" src="/Photos/photo3.jpeg">
</p>
 
 [](/Photos/photo3.jpeg)  

# Summary  

## Future Work:  
* Use classes in workflow 

# Resources: 
* Flatiron School Curriculum 
* DataDriven.org 
* Unsplash 

# [Blog Post](https://medium.com/@kristendavis27/graphing-feature-importance-with-scatter-polar-plots-72e9d0cb1d9c)
