# Project 2 - Ames Housing Data and Kaggle Challenge

## Problem Statement
Propnex is a real estate company in Ames, Iowa. Being part of the data science team, I have been tasked to create regression models based on Ames Housing dataset from 2006 to 2010 to predict the price of house at sale in Ames, Iowa. <br><br>
More than often, homeowners are unfamiliar with how much can their house fetch in the market and what are the factors that influence the house sale price. I hope to provide clear insights so they can have an estimate of their house price. And if the sale price is not ideal, they can consider to make improvements to their house based on the key variables.<br><br>
I will use different regression models and see which model is the best at predicting the sale price and narrow down the key variables that homeowners should look out for. I will evaluate my model based on the r2 score for both train and validation date and cross validation score. If all 3 scores are high (above 0.9) and very close to each other, I will deem it as a good model.<br><br>
By selecting the best model, it will be helpful to homeowners to understand:
- what are the key features that can positively or negatively affect the sale price
- given the current state of their house, what is the expected sale price


### Overview
I will use linear, lasso and ridge regression modelling to predict the sale price in the test data and will select the model with the highest r2 score and cross validation score as the final modelling.<br>


### Summary
There are many variables in the dataset and I will look at the key variables that positively or negatively influence the Sale Price by using regression models. By targeting at the variables that have large/significant impact on the sale price, this will allow homeowners to understand what factors influence sale price and if there's anything they can do to improve it.
<br>


### Conclusion and recommendations
Based on my modelling above, I conclude that Lasso and Ridge regression after feature engineering is the best model to predict the sale price. 
Lasso helps to eliminate unimportant features and narrow down the features that do affect sale price, with coefficients stating how much do they positively and negatively influence sale price. <br><br>
Creation of new features help to provide insights whether interaction features might influence sale price and in this case, it did influence sale price. <br><br>
Looking at the Ridge coefficients dataframe, I can conclude that these are some of the key factors that will influence sale price:
1. Ground living area * Overall Quality
2. Ground living area * Kitchen Quality
3. Overall Quality * External Quality
4. Neighbourhood - Northridge Heights and Stone Brooke
5. Basement Type 1 finished square feet
6. Total Basement square feet

These are the variables that will negatively affect sale price:
1. Roof Material - ClayTile
2. Misc Feature - Elevator

I would recommend home buyers to improve the house overall quality, kitchen quality and exterior quality by doing minor renovation or painting to their house if the current state is not of a good quality. <br><br>
Based on the dataframe above, the top 3 variables are interaction features which mean a combination of Ground living area & Overall Quality, Ground living area & Kitchen Quality and Overall Quality & External Quality respectively have a significant positive impact on the sale price. For Ground living area, there's not much homeowners can do about it since the area/square feet is fixed but at least it gives homeowner a base idea of how much their house will cost. <br><br>
If minor renovation or painting that does not cost much can do the trick, homeowners can consider doing that in order to fetch a higher price in the market. Otherwise, if the cost of renovation or painting outweight the increase in sale price of the house, home owners might want to reconsider this option.<br><br>
For houses in Northridge Heights and Stone Brooke neighbourhoods, homeowners can expect their house to fetch a higher price. 
Northridge Heights and is a family-friendly neighbourfood with many amenities nearby that is within walking distance. Also, it is in the thriving Gilbert School District.<br>
Stone Brooke is located nearby of the Iowa State University campus and a shopping mall. And all residents are free to use amenities such as swimming pool and club house and there's even amonthly potluck lunch as well. 
Perhaps, the characteristics of these two neighbourhood help to influence the sale price as the amenities are pretty attractive for both individuals and families.

Looking at variables that affect the sale price negatively that is Roof material made of ClayTile and Misc Feature - Elevator, homeowners that have either these two variables would need to be prepared that their house would not be able to fetch a good sale price. Based on the dataset, it seems like very few houses in Ames has roof material made of ClayTile or has an elevator as there's only one house that has each variable so perhaps most home owners would not have to worry about. But in the event that if the house has either variable, they would need to take note the variable would hurt the sale price. They may want to consider changing the roof material / removing the elevator for a better sale price depending on the difference in sale price it can fetch.

For Propnex, the company can use the model to provide consulting services to advise homeowners what's the sale price of their house and recommend the key areas they can focus on to increase sale price. 

To further improve my models, I will consider removing the extreme outliers and create more interaction features to see if the r2 score and cross validation score can be further increased and be of very close values. 


