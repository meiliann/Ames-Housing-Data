# Project 2 - Ames Housing Data and Kaggle Challenge

## Problem Statement
Propnex is a real estate company in Ames, Iowa. Being part of the data science team, I have been tasked to create regression models based on Ames Housing dataset from 2006 to 2010 to predict the price of house at sale in Ames, Iowa. <br><br>
More than often, homeowners are unfamiliar with how much can their house fetch in the market and what are the factors that influence the house sale price. I hope to provide clear insights so they can have an estimate of their house price. And if the sale price is not ideal, they can consider to make improvements to their house based on the key variables.<br><br>
I will use different regression models and see which model is the best at predicting the sale price and narrow down the key variables that homeowners should look out for. I will evaluate my model based on the r2 score for both train and validation date and cross validation score. If all 3 scores are high (above 0.9) and very close to each other, I will deem it as a good model.<br><br>
By selecting the best model, it will be helpful to homeowners to understand:
- what are the key features that can positively or negatively affect the sale price
- given the current state of their house, what is the expected sale price


### Data dictionary
---
   ID: Observation number<br><br>
   PID: Parcel identification number  - can be used with city web site for parcel review. <br><br>
   MS_SubClass: Identifies the type of dwelling involved in the sale.<br>
       - 20 1-STORY 1946 & NEWER ALL STYLES<br>
       - 30 1-STORY 1945 & OLDER<br>
       - 40 1-STORY W/FINISHED ATTIC ALL AGES<br>
       - 45 1-1/2 STORY - UNFINISHED ALL AGES<br>
       - 50 1-1/2 STORY FINISHED ALL AGES<br>
       - 60 2-STORY 1946 & NEWER<br>
       - 70 2-STORY 1945 & OLDER<br>
       - 75 2-1/2 STORY ALL AGES<br>
       - 80 SPLIT OR MULTI-LEVEL<br>
       - 85 SPLIT FOYER<br>
       - 90 DUPLEX - ALL STYLES AND AGES<br>
       - 120 1-STORY PUD (Planned Unit Development) - 1946 & NEWER<br>
       - 150 1-1/2 STORY PUD - ALL AGES<br>
       - 160 2-STORY PUD - 1946 & NEWER<br>
       - 180 PUD - MULTILEVEL - INCL SPLIT LEV/FOYER<br>
       - 190 2 FAMILY CONVERSION - ALL STYLES AND AGES<br><br>
   MS_Zoning: Identifies the general zoning classification of the sale.<br>
       - A Agriculture<br>
       - C Commercial<br>
       - FV Floating Village Residential<br>
       - I Industrial<br>
       -  RH Residential High Density<br>
       - RL Residential Low Density<br>
       - RP Residential Low Density Park<br>
       - RM Residential Medium Density<br><br>
   Lot_Frontage: Linear feet of street connected to property<br><br>
   Lot_Area: Lot size in square feet<br><br>
   Street: Type of road access to property<br>
       - Grvl Gravel<br>
       - Pave Paved<br><br>
   Alley: Type of alley access to property<br>
       - Grvl Gravel<br>
       - Pave Paved<br>
       - None No alley access<br>
   Lot_Shape: General shape of property<br><br>
       - Reg Regular<br>
       - IR1 Slightly irregular<br>
       - IR2 Moderately Irregular<br>
       - IR3 Irregular<br><br>
   Land_Contour: Flatness of the property<br>
       - Lvl Near Flat/Level<br>
       - Bnk Banked - Quick and significant rise from street grade to building<br>
       - HLS Hillside - Significant slope from side to side<br>
       - Low Depression<br>
   Utilities: Type of utilities available<br>
       - AllPub All public Utilities (E,G,W,& S)<br>
       - NoSewr Electricity, Gas, and Water (Septic Tank)<br>
       - NoSeWa Electricity and Gas Only<br>
       - ELO Electricity only<br><br>
   Lot_Config: Lot configuration<br>
       - Inside Inside lot<br>
       - Corner Corner lot<br>
       - CulDSac Cul-de-sac<br>
       - FR2 Frontage on 2 sides of property<br>
       - FR3 Frontage on 3 sides of property<br><br>
   Land_Slope: Slope of property<br>
       - Gtl Gentle slope<br>
       - Mod Moderate Slope<br>
       - Sev Severe Slope<br><br>
   Neighborhood: Physical locations within Ames city limits<br>
       - Blmngtn Bloomington Heights<br>
       - Blueste Bluestem<br>
       - BrDale Briardale<br>
       - BrkSide Brookside<br>
       - ClearCr Clear Creek<br>
       - CollgCr College Creek<br>
       - Crawfor Crawford<br>
       - Edwards Edwards<br>
       - Gilbert Gilbert<br>
       - GrnHill Green Hill<br>
       - IDOTRR Iowa DOT and Rail Road<br>
       - MeadowV Meadow Village<br>
       - Mitchel Mitchell<br>
       - Names North Ames<br>
       - NoRidge Northridge<br>
       - NPkVill Northpark Villa<br>
       - NridgHt Northridge Heights<br>
       - NWAmes Northwest Ames<br>
       - OldTown Old Town<br>
       - SWISU South & West of Iowa State University<br>
       - Sawyer Sawyer<br>
       - SawyerW Sawyer West<br>
       - Somerst Somerset<br>
       - StoneBr Stone Brook<br>
       - Timber Timberland<br>
       - Veenker Veenker<br><br>
   Condition_1: Proximity to main road or railroad<br>
       - Artery Adjacent to arterial street<br>
       - Feedr Adjacent to feeder street<br>
       - Norm Normal<br>
       - RRNn Within 200' of North-South Railroad<br>
       - RRAn Adjacent to North-South Railroad<br>
       - PosN Near positive off-site feature--park, greenbelt, etc.<br>
       - PosA Adjacent to postive off-site feature<br>
       - RRNe Within 200' of East-West Railroad<br>
       - RRAe Adjacent to East-West Railroad<br><br>
   Condition_2: Proximity to main road or railroad (if a second is present)<br>
       - Artery Adjacent to arterial street<br>
       - Feedr Adjacent to feeder street<br>
       - Norm Normal<br>
       - RRNn Within 200' of North-South Railroad<br>
       - RRAn Adjacent to North-South Railroad<br>
       - PosN Near positive off-site feature--park, greenbelt, etc.<br>
       - PosA Adjacent to postive off-site feature<br>
       - RRNe Within 200' of East-West Railroad<br>
       - RRAe Adjacent to East-West Railroad<br><br>
   Bldg_Type: Type of dwelling<br>
       - 1Fam Single-family Detached<br>
       - 2FmCon Two-family Conversion; originally built as one-family dwelling<br>
       - Duplx Duplex<br>
       - TwnhsE Townhouse End Unit<br>
       - TwnhsI Townhouse Inside Unit<br><br>
   House_Style: Style of dwelling<br>
       - 1Story One story<br>
       - 1.5Fin One and one-half story: 2nd level finished<br>
       - 1.5Unf One and one-half story: 2nd level unfinished<br>
       - 2Story Two story<br>
       - 2.5Fin Two and one-half story: 2nd level finished<br>
       - 2.5Unf Two and one-half story: 2nd level unfinished<br>
       - SFoyer Split Foyer<br>
       - SLvl Split Level<br><br>
   Overall_Qual: Overall material and finish quality<br>
       - 10 Very Excellent<br>
       - 9 Excellent<br>
       - 8 Very Good<br>
       - 7 Good<br>
       - 6 Above Average<br>
       - 5 Average<br>
       - 4 Below Average<br>
       - 3 Fair<br>
       - 2 Poor<br>
       - 1 Very Poor<br><br>
   Overall_Cond: Overall condition rating<br>
       - 10 Very Excellent<br>
       - 9 Excellent<br>
       - 8 Very Good<br>
       - 7 Good<br>
       - 6 Above Average<br>
       - 5 Average<br>
       - 4 Below Average<br>
       - 3 Fair<br>
       - 2 Poor<br>
       - 1 Very Poor<br><br>
   Year_Built: Original construction date<br><br>
   Year_Remod_Add: Remodel date (same as construction date if no remodeling or additions)<br><br>
   Roof_Style: Type of roof<br>
       - Flat Flat<br>
       - Gable Gable<br>
       - Gambrel Gabrel (Barn)<br>
       - Hip Hip<br>
       - Mansard Mansard<br>
       - Shed Shed<br><br>
   Roof_Matl: Roof material<br>
       - ClyTile Clay or Tile<br>
       - CompShg Standard (Composite) Shingle<br>
       - Membran Membrane<br>
       - Metal Metal<br>
       - Roll Roll<br>
       - Tar&Grv Gravel & Tar<br>
       - WdShake Wood Shakes<br>
       - WdShngl Wood Shingles<br><br>
   Exterior_1st: Exterior covering on house<br>
       - AsbShng Asbestos Shingles<br>
       - AsphShn Asphalt Shingles<br>
       - BrkComm Brick Common<br>
       - BrkFace Brick Face<br>
       - CBlock Cinder Block<br>
       - CemntBd Cement Board<br>
       - HdBoard Hard Board<br>
       - ImStucc Imitation Stucco<br>
       - MetalSd Metal Siding<br>
       - Other Other<br>
       - Plywood Plywood<br>
       - PreCast PreCast<br>
       - Stone Stone<br>
       - Stucco Stucco<br>
       - VinylSd Vinyl Siding<br>
       - Wd Sdng Wood Siding<br>
       - WdShing Wood Shingles<br><br>
   Exterior_2nd: Exterior covering on house (if more than one material)<br>
       - AsbShng Asbestos Shingles<br>
       - AsphShn Asphalt Shingles<br>
       - BrkComm Brick Common<br>
       - BrkFace Brick Face<br>
       - CBlock Cinder Block<br>
       - CemntBd Cement Board<br>
       - HdBoard Hard Board<br>
       - ImStucc Imitation Stucco<br>
       - MetalSd Metal Siding<br>
       - Other Other<br>
       - Plywood Plywood<br>
       - PreCast PreCast<br>
       - Stone Stone<br>
       - Stucco Stucco<br>
       - VinylSd Vinyl Siding<br>
       - Wd Sdng Wood Siding<br>
       - WdShing Wood Shingles<br><br>
   Mas_Vnr_Type: Masonry veneer type<br>
       - BrkCmn Brick Common<br>
       - BrkFace Brick Face<br>
       - CBlock Cinder Block<br>
       - None None<br>
       - Stone Stone<br><br>
   Mas_Vnr_Area: Masonry veneer area in square feet<br>
   Exter_Qual: Exterior material quality<br>
       - Ex Excellent<br>
       - Gd Good<br>
       - TA Average/Typical<br>
       - Fa Fair<br>
       - Po Poor<br><br>
   Exter_Cond: Present condition of the material on the exterior<br>
       - Ex Excellent<br>
       - Gd Good<br>
       - TA Average/Typical<br>
       - Fa Fair<br>
       - Po Poor<br><br>
   Foundation: Type of foundation<br>
       - BrkTil Brick & Tile<br>
       - CBlock Cinder Block<br>
       - PConc Poured Contrete<br>
       - Slab Slab<br>
       - Stone Stone<br>
       - Wood Wood<br>
   Bsmt_Qual: Height of the basement<br>
       - Ex Excellent (100+ inches)<br>
       - Gd Good (90-99 inches)<br>
       - TA Typical (80-89 inches)<br>
       - Fa Fair (70-79 inches)<br>
       - Po Poor (<70 inches)<br>
       - None No Basement<br><br>
   Bsmt_Cond: General condition of the basement<br>
       - Ex Excellent<br>
       - Gd Good<br>
       - TA Typical - slight dampness allowed<br>
       - Fa Fair - dampness or some cracking or settling<br>
       Po Poor - Severe cracking, settling, or wetness<br>
       None No Basement<br><br>
   Bsmt_Exposure: Walkout or garden level basement walls<br>
       - Gd Good Exposure<br>
       - Av Average Exposure (split levels or foyers typically score average or above)<br>
       - Mn Mimimum Exposure<br>
       - No No Exposure<br>
       - None No Basement<br><br>
   BsmtFinType1: Quality of basement finished area<br>
       - GLQ Good Living Quarters<br>
       - ALQ Average Living Quarters<br>
       - BLQ Below Average Living Quarters<br>
       - Rec Average Rec Room<br>
       - LwQ Low Quality<br>
       - Unf Unfinshed<br>
       - None No Basement<br>
   BsmtFin_SF_1: Type 1 finished square feet<br>
   BsmtFin_Type_2: Quality of second finished area (if present)<br>
       - GLQ Good Living Quarters<br>
       - ALQ Average Living Quarters<br>
       - BLQ Below Average Living Quarters<br>
       - Rec Average Rec Room<br>
       - LwQ Low Quality<br>
       - Unf Unfinshed<br>
       - None No Basement<br><br>
   BsmtFin_SF_2: Type 2 finished square feet<br><br>
   Bsmt_Unf_SF: Unfinished square feet of basement area<br><br>
   Total_Bsmt_SF: Total square feet of basement area<br><br>
   Heating: Type of heating<br>
       - Floor Floor Furnace<br>
       - GasA Gas forced warm air furnace<br>
       - GasW Gas hot water or steam heat<br>
       - Grav Gravity furnace<br>
       - OthW Hot water or steam heat other than gas<br>
       - Wall Wall furnace<br><br>
   Heating_QC: Heating quality and condition<br>
       - Ex Excellent<br>
       - Gd Good<br>
       - TA Average/Typical<br>
       - Fa Fair<br>
       - Po Poor<br><br>
   Central_Air: Central air conditioning<br>
       - N No<br>
       - Y Yes<br>
   Electrical: Electrical system<br>
       - SBrkr Standard Circuit Breakers & Romex<br>
       - FuseA Fuse Box over 60 AMP and all Romex wiring (Average)<br>
       - FuseF 60 AMP Fuse Box and mostly Romex wiring (Fair)<br>
       - FuseP 60 AMP Fuse Box and mostly knob & tube wiring (poor)<br>
       - Mix Mixed<br>
   1st_Flr_SF: First Floor square feet<br><br>
   2nd_Flr_SF: Second floor square feet<br><br>
   Low_Qual_Fin_SF: Low quality finished square feet (all floors)<br><br>
   Gr_Liv_Area: Above grade (ground) living area square feet<br><br>
   Bsmt_Full_Bath: Basement full bathrooms<br><br>
   Bsmt_Half_Bath: Basement half bathrooms<br><br>
   Full_Bath: Full bathrooms above grade<br><br>
   Half_Bath: Half baths above grade<br>
   Bedroom: Number of bedrooms above basement level<br><br>
   Kitchen: Number of kitchens<br><br>
   Kitchen_Qual: Kitchen quality<br><br>
       - Ex Excellent<br>
       - Gd Good<br>
       - TA Typical/Average<br>
       - Fa Fair<br>
       - Po Poor<br><br>
   TotRms_Abv_Grd: Total rooms above grade (does not include bathrooms)<br>
   Functional: Home functionality rating<br>
       - Typ Typical Functionality<br>
       - Min1 Minor Deductions 1<br>
       - Min2 Minor Deductions 2<br>
       - Mod Moderate Deductions<br>
       - Maj1 Major Deductions 1<br>
       - Maj2 Major Deductions 2<br>
       - Sev Severely Damaged<br>
       - Sal Salvage only<br>
   Fireplaces: Number of fireplaces<br><br>
   Fireplace_Qu: Fireplace quality<br>
       - Ex Excellent - Exceptional Masonry Fireplace<br>
       - Gd Good - Masonry Fireplace in main level<br>
       - TA Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement<br>
       - Fa Fair - Prefabricated Fireplace in basement<br>
       - Po Poor - Ben Franklin Stove<br>
       - None No Fireplace<br><br>
   Garage_Type: Garage location<br>
       - 2Types More than one type of garage<br>
       - Attchd Attached to home<br>
       - Basment Basement Garage<br>
       - BuiltIn Built-In (Garage part of house - typically has room above garage)<br>
       - CarPort Car Port<br>
       - Detchd Detached from home<br>
       - None No Garage<br>
   Garage_Yr_Blt: Year garage was built<br><br>
   Garage_Finish: Interior finish of the garage<br><br>
       - Fin Finished<br>
       - RFn Rough Finished<br>
       - Unf Unfinished<br>
       - None No Garage<br>
   Garage_Cars: Size of garage in car capacity<br><br><br><br>
   Garage_Area: Size of garage in square feet<br><br>
   Garage_Qual: Garage quality<br>
       - Ex Excellent<br>
       - Gd Good<br>
       - TA Typical/Average<br>
       - Fa Fair<br>
       - Po Poor<br>
       - None No Garage<br><br>
   Garage_Cond: Garage condition<br>
       - Ex Excellent<br>
       - Gd Good<br>
       - TA Typical/Average<br>
       - Fa Fair<br>
       - Po Poor<br>
       - None No Garage<br><br>
   Paved_Drive: Paved driveway<br>
       - Y Paved<br>
       - P Partial Pavement<br>
       - N Dirt/Gravel<br>
   Wood_Deck_SF: Wood deck area in square feet<br><br>
   Open_Porch_SF: Open porch area in square feet<br><br>
   Enclosed_Porch: Enclosed porch area in square feet<br><br>
   3Ssn_Porch: Three season porch area in square feet<br><br>
   Screen_Porch: Screen porch area in square feet<br><br>
   Pool_Area: Pool area in square feet<br><br>
   Pool_QC: Pool quality<br>
       - Ex Excellent<br>
       - Gd Good<br>
       - TA Average/Typical<br>
       - Fa Fair<br>
       None No Pool<br><br>
   Fence: Fence quality<br>
       - GdPrv Good Privacy<br>
       - MnPrv Minimum Privacy<br>
       - GdWo Good Wood<br>
       - MnWw Minimum Wood/Wire<br>
       - None No Fence<br><br>
   Misc_Feature: Miscellaneous feature not covered in other categories<br>
       - Elev Elevator<br>
       - Gar2 2nd Garage (if not described in garage section)<br>
       - Othr Other<br>
       - Shed Shed (over 100 SF)<br>
       - TenC Tennis Court<br>
       - None None<br><br>
   Misc_Val: $Value of miscellaneous feature<br>
   Mo_Sold: Month Sold<br><br>
   Yr_Sold: Year Sold<br><br>
   Sale_Type: Type of sale<br>
       - WD Warranty Deed - Conventional<br>
       - CWD Warranty Deed - Cash<br>
       - VWD Warranty Deed - VA Loan<br>
       - New Home just constructed and sold<br>
       - COD Court Officer Deed/Estate<br>
       - Con Contract 15% Down payment regular terms<br>
       - ConLw Contract Low Down payment and low interest<br>
       - ConLI Contract Low Interest<br>
       - ConLD Contract Low Down<br>
       - Oth Other<br><br>
   SalePrice - the property's sale price in dollars. This is the variable that I want to predict.
   
   
<br>

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


