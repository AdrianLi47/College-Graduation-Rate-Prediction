# College-Graduation-Rate-Prediction
## 🗒️Project Overwiew:
Analysis of data from 777 US colleges and universities, predicting student graduation rates, using Linear Regression and Random Forest algorithms — from data cleaning, through EDA, to hyperparameter tuning with GridSearchCV.
## Menu
- [Dataset](#dataset)
- [Methodology](#methodology)
## 📊Dataset
### Variables:
- Private - Whether the institution is private 
- Apps - Number of applications received 
- Accept - Number of applicants accepted 
- Enroll - Number of new students enrolled 
- Top10perc - % of new students from the top 10% of their high school class 
- Top25perc - % of new students from the top 25% of their high school class 
- F.Undergrad - Number of full-time undergraduates 
- P. Undergrad - Number of part-time undergraduates 
- Outstate - Out-of-state tuition 
- Room.Board - Room and board costs 
- Books - Estimated book costs 
- Personal - Estimated personal spending 
- PhD - % of faculty with a Ph.D. 
- Terminal - % of faculty with a terminal degree 
- S.F.Ratio - Student/faculty ratio 
- perc.alumni - % of alumni who donate 
- Expend - Instructional expenditure per student
- **Grad_Rate** - **Target variable — graduation rate** 
## 🧩Methodology
- Data cleaning — dropping missing rows - (dropna()), clipping unrealistic values above 100% for Grad.Rate, PhD, Terminal - (.clip(upper=100)).
- Exploratory Data Analysis (EDA) — correlation matrix between predictors and the target variable.
- Data split — 80% train / 20% test, random_state=327734.
- Preprocessing — standardization of numeric variables (StandardScaler), encoding of the Private variable (OneHotEncoder(drop='first')).
- Modeling — Linear Regression and Random Forest (100 / 200 trees, then hyperparameter tuning via GridSearchCV).
### 💡Algorithms used
- Model	Parameters
- Linear Regression	scikit-learn default settings
- Random Forest (v1)	n_estimators=100, random_state=327734
- Random Forest (v2)	n_estimators=200, random_state=327734
- Random Forest after GridSearchCV	n_estimators=300, max_depth=20, max_features='sqrt', min_samples_split=2 (search grid: cv=5, scoring='r2', n_jobs=-1)
## Results
| Model | R² | MAE | RMSE | 
| ------------- | ------------- | ------------- | ------------- |
| Linear Regression | 44.28% | 9.29 | 12.01 | 
| Random Forest (after GridSearchCV) | 41.74% | 9.24 | 12.29 | 

			
