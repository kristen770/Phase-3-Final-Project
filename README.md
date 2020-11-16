# Phase-3-Final-Project  
ByL Kristen Davis  

<p align="center">
  <img width="600" height="600" src="/Photos/newplot.png">
</p>
 
 [](/Photos/newplot.png)   

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
Below is a brief summary of the extensive EDA preformed on this dataset. In conjuction with modeling, these questions helped be understand the 'picture' of functioning and non functioning water wells in Tanzania.   

* Based on a vanilla Random Forest the most important features for predicting construction year are: scheme_name, installer, scheme_management, permit

* In general newer pumps are working regardless of pay structure
* Lower district codes have a higher number of overall wells, non functional wells, and wells that need repair

* If you consider the functional needs repair as functional all basins have more functioning wells than no functioning.
* Lake Victoria has the closest ratio: 5100 functional to 4159 non-functional
* Lake Nyassa and Rufiji have the largest percentage of wells that are functioning
* Lake Rukwa has the smallest number of overall wells 

 * There are 289 functional wells in Lgosi (Ward). this is significantly more than any other ward. The majority of wards have 50 wells in any given status group.
 
 * Pangani basin has the highest number of pumps producing insufficent amounts of water.
 
 * Njombe has 447 water pumps funded by UNICEF
 * 80% of water pumps are functioning in Njombe in the Rufiji
 * The majority of the funtional water pumps were funded by UNICEF / installed by DWE / Had a publice meeting / opperated by WUA / VWC, were constructed after 2000, are gravity / pay monthly / soft / enough / spring in a communal standpipe
 
 * Funded by the Government of Tanzania (2131 of 19640 / 11% overall)
* 649 have a GPS Height between 0 and 50 (largest grouping)
* Largest grouping in Rufiji but well distributed across most basins (Lake Rukwa/ Ruvuma smallest) n
* Most in the Iringa Region
* Most (1873) in LGA Njombe
* 87% had a public meeting
* 59% of the wells are managed by VWC
* 67% of the wells have a permit
* 55% of the wells use gravity as their extraction type
* Most never pay with second most common pay structure being "pay per bucket" and "pay monthly"
* 7807 wells have springs as their water source
* 77% of wells have ground water as their water source
* 67% of wells have a communal standpipe

* The majority of wells that are functioning well with sufficient water are funded by the Government of Tanzania but if you look at all the wells the Government of Tanzania is making they are making more bad wells than good wells. Not good.

* Largest group of government built water pumps in Pangani basin
* 7831 wells were constructed after a public meeting, presumably
* 6395 wells are managed by VWC
* 67775 wells have a permit
* 5550 wells are gravity extraction types
* 4723 wells are never pay
* 8041 wells are soft good quality. The largest bad water quality is salty 628
* 4705 wells have enough water for their region, 3025 do not have enough
* 4014 wells have springs as their water source
* 6862 wells have ground water as their water source 
* 5264 have communal standpipes as their water source

* The vast majority of pumps that the government funds they also install

* The biggest issue with not soft water is the saltiness

* There are more nonfunctioning than functioning wells (2411 vs 2220)
* The majority of the salty water wells are in Wami/ Ruvu (1217) the least in Lake Nyasa (13)
* A large group are submersive
* Machine dbh(2234) / shallow well (2274) - groundwater

* The vast majority of water is good ('soft') regardless of payment type.

* Abandoned floride counts = 180 functional 104 non functional 72 functional needs repair
* Abandoned salty counts = 174 functional 93 non functional
* The majority of unknown quality pumps are from the internal basin
* There are more nonfunctiong salty water pumps than functioning (2411 salty non-functioning / 2220 satly functioning)

* Boreholes produce enough good water (29.901k)

* It is almost a 50/50 split if a communal standpipe is working or not. If it is never pay much higher precentages of funtioning pipes in pay per bucket and pay monthly models (16%/ 16%)
* 12306 - 46% / 45% / 9% 

* 6052 of 22824 total water pumps that are not functioning also are 'dry' 
* 37 of 4317 total water pumps that are in need of repair also are 'dry' 

# Modeling  
<p align="center">
  <img width="600" height="250" src="/Photos/photo4.jpeg">
</p>
 
 [](/Photos/photo4.jpeg) 
 
 Modeling this data provided an oppurtunity to explore multiple modeling techniques: KNN, Logistic Regression, Random Forests and other ensemble methods. Through out modeling my goal was to identify the model, feature and parameters that would best predict the status of well in Tanzania. Within each model I took approach, enumerating on a variety of models and data processing techniques. Below are the testing accuracy scores for each of my models, as well as a detailed summary of my final model. 
 
 * Single Tree Vanilla Processing - Testing Accuracy for Decision Tree Classifier: 70.88%   
 * Bagged Decision Tree Vanilla Processing - Testing Accuracy for Decision Tree Classifier: 71.27% 
 * Random Forest Vanilla Processing - Testing Accuracy for Random Forest Classifier: 55.47% 
 * Random forest Grid Search: Testing Accuracy for Random Forest Classifier: 74.14%  

## Final Model 
<p align="center">
  <img width="600" height="250" src="/Photos/photo3.jpeg">
</p>
 
 [](/Photos/photo3.jpeg)  

# Summary  

## Future Work:  
* Use classes in workflow  
* Use sklearn pipelines throught out modeling 

# Resources: 
* Flatiron School Curriculum 
* DataDriven.org 
* Unsplash 

# [Blog Post](https://medium.com/@kristendavis27/graphing-feature-importance-with-scatter-polar-plots-72e9d0cb1d9c)
