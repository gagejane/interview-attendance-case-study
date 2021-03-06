# Interview Attendance Case Study

#### Team: Ryan Curry, Emily Quigley, Millie Smith, Jane Stout


### Working Agenda
 10:00 - 10:30 Team meeting to strategize
<br>
 10:30 - 12:00 Cleaning + EDA
<br>
 12:00 - 12:20 Team meeting to share results
<br>
 12:30 -  3:00 Modeling and analysis
<br>
 3:00  -  4:00 Markdown and GitHub

### Data Cleaning
1. Broke columns and tasks out between the 4 of us for efficient cleaning
2. Retitled column names by removing spaces and creating shorter names
3. Explored data by using .unique() and .value_counts()
4. Simplified values by combining ["No","no","not yet"] into a single value
5. Converted to binary including dummy variables for columns such as location

### Data Merging
To combine our separate files with the specific columns we worked on, we all exported our columns to our own CSV and then combined these together into a new dataframe. Upon merging, we dropped our NaN values. Figure 1 provides a visualization of NaN values across variables.
<br>
<br>

![msno.png](msno.png)

**Figure 1. Missing Values**

### Modeling
1. Split into train and test
2. Stratified with respect to y in order to account for any class imbalance (63:36)
3. Started with Random Forest Classifier
4. Looked at the influence scores of the feature (see Figure 2)

![](Infl_scores2.png)

**Figure 2. Influence scores**

5. Fit AdaBoost & Gradient Boosting Classifier to compare models
6. Calculated metrics for all models (see Table 1)
We found more false positives than false negatives. That is, we predicted more people were going to show up than actually did.


|Metrics   |Score   |
|---|---|
|RFC OOB | 0.73 |
|RFC Test | 0.72 |
|ABC Test | 0.70 |
|Gradient Boosting Regressor Test | 0.74 |
|RFC precision | 0.74 |
|ABC precision | 0.74 |
|Gradient Boosting Regressor precision | 0.75 |
|RFC recall | 0.92 |
|ABC recall | 0.89 |
|Gradient Boosting Regressor recall | 0.92 |
**Table 1. Metrics scores**
<br>

### Future Work
* Tune hyperparameters
* Use K-Nearest Neighbors for NaN values
