import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import warnings
warnings.simplefilter(action="ignore")

def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]


    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

df = pd.read_csv('Telco-Customer-Churn.csv')

def first_prep(dataframe, head=5):
    pd.set_option('display.max_columns', None)  # *
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 170)

    print("##################### Shape #####################")
    print(dataframe.shape)

    print("##################### Types #####################")
    print(dataframe.dtypes)

    print("##################### Head #####################")
    print(dataframe.head(head))

    print("##################### NA #####################")
    print(dataframe.isnull().sum())

    print("##################### Describe #####################")
    print(dataframe.describe().T)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

    df['Churn'].replace({'Yes': 1, 'No': 0}, inplace=True)
    df.isnull().sum()
    df.dropna(axis=0, inplace=True)

    no_int_pars = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                   'StreamingTV', 'StreamingMovies']

    [df[col].replace({'Yes': 1, 'No': 0, 'No internet service': 0}, inplace=True) for col in no_int_pars]
    df['MultipleLines'].replace({'Yes': 1, 'No': 0, 'No phone service': 0}, inplace=True)


first_prep(df)
# We have 7043 rows and 21 columns
# There is no missing values on dataset

###########################################
# Numerical Variable Analysis with Graphs #
###########################################

num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
print(df[num_cols].describe(quantiles).T)

def num_plot(data, cat_length=16, remove=["Id"], hist_bins=12, figsize=(20, 4)):
    num_cols = [col for col in data.columns if data[col].dtypes != "O"
                and len(data[col].unique()) >= cat_length]

    if len(remove) > 0:
        num_cols = list(set(num_cols).difference(remove))

    for i in num_cols:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        data.hist(str(i), bins=hist_bins, ax=axes[0])
        data.boxplot(str(i), ax=axes[1], vert=False);
        try:
            sns.kdeplot(np.array(data[str(i)]))
        except:
            ValueError

        axes[1].set_yticklabels([])
        axes[1].set_yticks([])
        axes[0].set_title(i + " | Histogram")
        axes[1].set_title(i + " | Boxplot")
        axes[2].set_title(i + " | Density")
        plt.show()

num_plot(df)

# When I examine the numerical variables, there is no problem in their distribution.
# Therefore, it is unnecessary to use operations such as log transformation or removing outliers.


#############################################
# Categorical Variable Analysis with Graphs #
#############################################

cat_cols = ['gender', 'SeniorCitizen','Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod']

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col, plot=True)

# When I analysed the categorical variables, I did not come across classes with rare percentages.
# Classes are generally evenly distributed.
# We have information about the variables, but analysing their effects on our target variable will yield more useful outputs



###############################################################
# The Effects Of Categorical Variables On The Target Variable #
###############################################################

def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)


########################
# Correlation Analysis #
########################

f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

# As we expected, TotalChargers is highly correlated with monthly wages and tenure.


############
# Encoding #
############
cat_cols, num_cols, cat_but_car = grab_col_names(df)
cat_cols.remove('Churn')


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)

df.head()
###############
# BASE MODELS #
###############

y = df["Churn"]
X = df.drop(["Churn", "customerID"], axis=1)

models = [('LR', LogisticRegression(random_state=2804)),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=2804)),
          ('RF', RandomForestClassifier(random_state=2804)),
          ('GBM', GradientBoostingClassifier(random_state=2804)),
          ('XGB', XGBClassifier(random_state=2804)),
          ("LightGBM", LGBMClassifier(random_state=2804))]
          #("CatBoost", CatBoostClassifier(verbose=False, random_state=2804))]


for name, model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")

first_results = {'LR': [{'ACC': 0.8045, 'AUC': 0.8427, 'F1': 0.5958}],
                'RF': [{'ACC': 0.7927, 'AUC': 0.8253, 'F1': 0.5548}],
                'GBM': [{'ACC': 0.8045, 'AUC': 0.8472, 'F1': 0.5871}],
                'LGBM': [{'ACC': 0.7967, 'AUC': 0.8361, 'F1': 0.5805}]}

# We have 4 candidate models for our project. LR, RF, GBM, LightGBM
# Firstly, we will analyse the important variables in these 4 models.
# In this way, we will decide which variables we need to focus on more.

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

rf_model = RandomForestClassifier().fit(X, y)
gbm_model= GradientBoostingClassifier().fit(X, y)
lgbm_model = LGBMClassifier().fit(X, y)

plot_importance(rf_model, X)
plot_importance(gbm_model, X)
plot_importance(lgbm_model, X)

# Top 5 rf = Charges, tenure, IntService_Fiber, PaymMethod_Electro
# Top5 gbm = tenure,IntService_Fiber,PayMeth_Electro,Contract_TwoYear,Charges
# Top5 lgbm = Charges, tenure, gender_Male, paymethod (generally Charges and ten)

#######################
# FEATURE ENGINEERING #
#######################

df = pd.read_csv('Telco-Customer-Churn.csv')

first_prep(df)

# We have 11 null values on TotalCharges. We can remove these lines as the number is low.
# We have already seen that there are no outliers in the numerical variable analysis, so we skip this step.


###   FEATURE EXTRACTION   ###
# tenure
df.loc[(df["tenure"] >= 0) & (df["tenure"] <= 12), "New_Tenure_Year"] = "0-1 Year"
df.loc[(df["tenure"] > 12) & (df["tenure"] <= 24), "New_Tenure_Year"] = "1-2 Year"
df.loc[(df["tenure"] > 24) & (df["tenure"] <= 36), "New_Tenure_Year"] = "2-3 Year"
df.loc[(df["tenure"] > 36), "New_Tenure_Year"] = "3+ Year"
cat_cols.append('New_Tenure_Year')


# total charge group
df['New_Total_Charge_Group'] = pd.qcut(df['TotalCharges'], 3, labels=['Low', 'Middle', 'High'])
cat_cols.append('New_Total_Charge_Group')

# Streaming Services
df['StreamingTV'].replace({'Yes': 1, 'No': 0, 'No internet service': 0}, inplace=True)
df['StreamingMovies'].replace({'Yes': 1, 'No': 0, 'No internet service': 0}, inplace=True)

df['New_Stream_Rate'] = df['StreamingTV'] + df['StreamingMovies']

# PaymentFlag
df["New_PayM_Auto_Flag"] = df["PaymentMethod"].apply(lambda x: 1 if x in ["Bank transfer (automatic)","Credit card (automatic)"] else 0)
df.groupby('New_PayM_Auto_Flag').agg({'Churn': 'mean'}).sort_values('Churn', ascending=False)

# Total Service
df['New_Total_Service'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity',
                                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                       'StreamingTV', 'StreamingMovies']]== 'Yes').sum(axis=1)

# Charge Per Service
df['New_Charge_Per_Service'] = df['MonthlyCharges'] / (df['New_Total_Service'] + 1)
df['New_Charge_Total_Per_Service'] = df['TotalCharges'] / (df['New_Total_Service'] + 1)

# IntService
df['New_Int_TenureYear_Mean'] = df.groupby(['InternetService', 'New_Tenure_Year'])['Churn'].transform('mean')
df['New_Int_PayM_Mean'] = df.groupby(['InternetService', 'PaymentMethod'])['Churn'].transform('mean')
df['New_Int_Contract_Mean'] = df.groupby(['InternetService', 'Contract'])['Churn'].transform('mean')
df['New_Int_TotalCharge_Mean'] = df.groupby(['InternetService', 'New_Total_Charge_Group'])['Churn'].transform('mean')

# Contract
df['New_Contract_TenureYear_Mean'] = df.groupby(['Contract', 'New_Tenure_Year'])['Churn'].transform('mean')
df['New_Contract_PayM_Mean'] = df.groupby(['Contract', 'PaymentMethod'])['Churn'].transform('mean')
df['New_Contract_SenCiti_Mean'] = df.groupby(['Contract', 'SeniorCitizen'])['Churn'].transform('mean')
df['New_Contract_TotalCharge_Mean'] = df.groupby(['Contract', 'New_Total_Charge_Group'])['Churn'].transform('mean')

# Fiber Users
df.loc[(df['InternetService'] == 'Fiber optic') & (df['Contract'] == 'Month-to-month'), 'New_Cust_Value'] = 'VeryHighCare'
df.loc[(df['InternetService'] == 'DSL') & (df['Contract'] == 'Month-to-month'), 'New_Cust_Value'] = 'HighCare'
df.loc[(df['InternetService'] == 'Fiber optic') & (df['Contract'] == 'One year'), 'New_Cust_Value'] = 'MiddleCare'
df.loc[(df['InternetService'] == 'No') & (df['Contract'] == 'Month-to-month'), 'New_Cust_Value'] = 'LowCare'
df['New_Cust_Value'].fillna('VeryLowCare', inplace=True)

df['New_Cust_Charge_Mean'] = df.groupby('New_Cust_Value')['TotalCharges'].transform('mean')
df['New_Cust_Charge_Tenure_Mean'] = df.groupby(['New_Cust_Value', 'New_Tenure_Year'])['TotalCharges'].transform('mean')
df['New_Cust_Churn_Tenure_Mean'] = df.groupby(['New_Cust_Value', 'New_Tenure_Year'])['Churn'].transform('mean')
df['New_Cust_Charge_Paym_Mean'] = df.groupby(['New_Cust_Value', 'PaymentMethod'])['TotalCharges'].transform('mean')
df['New_Cust_Charge_Churn_Mean'] = df.groupby(['New_Cust_Value', 'PaymentMethod'])['Churn'].transform('mean')


df.groupby(['InternetService', 'Contract']).agg({'Churn': 'mean'}).sort_values('Churn', ascending=False)


#######################
# PREP FOR THE MODELS #
#######################

cat_cols, num_cols, cat_but_car = grab_col_names(df)
cat_cols.remove('Churn')

df = one_hot_encoder(df, cat_cols, drop_first=True)


###############################
# HYPERPARAMATER OPTIMIZATION #
###############################

# RANDOM FORESTS #

rf_model = RandomForestClassifier(random_state=17)

rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)


cv_results = cross_validate(rf_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.8347
cv_results['test_f1'].mean()
# 0.5769
cv_results['test_roc_auc'].mean()
# 0.8647


# LightGBM #
lgbm_model = LGBMClassifier(random_state=28)

lgbm_params = {"learning_rate": [0.01, 0.1, 0.001],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=10, n_jobs=-1, verbose=True).fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 84.11
cv_results['test_f1'].mean()
# 57.04
cv_results['test_roc_auc'].mean()
# 0.8651


# GBM #
gbm_model = GradientBoostingClassifier(random_state=28)

gbm_params = {'learning_rate': [0.01, 1],
              'max_depth': [3, 5, 8],
              'n_estimators': [100, 500, 1000],
              'subsample': [0.5, 0.7, 1]}

gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(gbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 84.11
cv_results['test_f1'].mean()
# 57.04
cv_results['test_roc_auc'].mean()
# 87.72

# Best model is GradientBoostingClassifier with 87.72 auc score.

plot_importance(gbm_final, X)

# When we look our variable importance graph ;
# We see that 10 of the 15 most important variables are our New variables.

















df.head()