# Keras_API_Project
 I will be using a subset of the LendingClub DataSet obtained from [Kaggle](https://www.kaggle.com/wordsforthewise/lending-club).
 ## About LendingCLub
 LendingClub is a US peer-to-peer lending company, headquartered in San Francisco, California. It was the first peer-to-peer lender to register its offerings as securities with the Securities and Exchange Commission (SEC), and to offer loan trading on a secondary market. LendingClub is the world's largest peer-to-peer lending platform.
 
Given historical data on loans given out with information on whether or not the borrower defaulted (charge-off),I will build a model that can predict wether or not a borrower will pay back their loan. This way in the future when the firm gets a new potential customer they will be able to assess whether or not they are likely to pay back the loan. 

## Data Overview


There are many LendingClub data sets on Kaggle. Here is the information on this particular data set:


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LoanStatNew</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>loan_amnt</td>
      <td>The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>term</td>
      <td>The number of payments on the loan. Values are in months and can be either 36 or 60.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>int_rate</td>
      <td>Interest Rate on the loan</td>
    </tr>
    <tr>
      <th>3</th>
      <td>installment</td>
      <td>The monthly payment owed by the borrower if the loan originates.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>grade</td>
      <td>LC assigned loan grade</td>
    </tr>
    <tr>
      <th>5</th>
      <td>sub_grade</td>
      <td>LC assigned loan subgrade</td>
    </tr>
    <tr>
      <th>6</th>
      <td>emp_title</td>
      <td>The job title supplied by the Borrower when applying for the loan.*</td>
    </tr>
    <tr>
      <th>7</th>
      <td>emp_length</td>
      <td>Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.</td>
    </tr>
    <tr>
      <th>8</th>
      <td>home_ownership</td>
      <td>The home ownership status provided by the borrower during registration or obtained from the credit report. Our values are: RENT, OWN, MORTGAGE, OTHER</td>
    </tr>
    <tr>
      <th>9</th>
      <td>annual_inc</td>
      <td>The self-reported annual income provided by the borrower during registration.</td>
    </tr>
    <tr>
      <th>10</th>
      <td>verification_status</td>
      <td>Indicates if income was verified by LC, not verified, or if the income source was verified</td>
    </tr>
    <tr>
      <th>11</th>
      <td>issue_d</td>
      <td>The month which the loan was funded</td>
    </tr>
    <tr>
      <th>12</th>
      <td>loan_status</td>
      <td>Current status of the loan</td>
    </tr>
    <tr>
      <th>13</th>
      <td>purpose</td>
      <td>A category provided by the borrower for the loan request.</td>
    </tr>
    <tr>
      <th>14</th>
      <td>title</td>
      <td>The loan title provided by the borrower</td>
    </tr>
    <tr>
      <th>15</th>
      <td>zip_code</td>
      <td>The first 3 numbers of the zip code provided by the borrower in the loan application.</td>
    </tr>
    <tr>
      <th>16</th>
      <td>addr_state</td>
      <td>The state provided by the borrower in the loan application</td>
    </tr>
    <tr>
      <th>17</th>
      <td>dti</td>
      <td>A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income.</td>
    </tr>
    <tr>
      <th>18</th>
      <td>earliest_cr_line</td>
      <td>The month the borrower's earliest reported credit line was opened</td>
    </tr>
    <tr>
      <th>19</th>
      <td>open_acc</td>
      <td>The number of open credit lines in the borrower's credit file.</td>
    </tr>
    <tr>
      <th>20</th>
      <td>pub_rec</td>
      <td>Number of derogatory public records</td>
    </tr>
    <tr>
      <th>21</th>
      <td>revol_bal</td>
      <td>Total credit revolving balance</td>
    </tr>
    <tr>
      <th>22</th>
      <td>revol_util</td>
      <td>Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.</td>
    </tr>
    <tr>
      <th>23</th>
      <td>total_acc</td>
      <td>The total number of credit lines currently in the borrower's credit file</td>
    </tr>
    <tr>
      <th>24</th>
      <td>initial_list_status</td>
      <td>The initial listing status of the loan. Possible values are – W, F</td>
    </tr>
    <tr>
      <th>25</th>
      <td>application_type</td>
      <td>Indicates whether the loan is an individual application or a joint application with two co-borrowers</td>
    </tr>
    <tr>
      <th>26</th>
      <td>mort_acc</td>
      <td>Number of mortgage accounts.</td>
    </tr>
    <tr>
      <th>27</th>
      <td>pub_rec_bankruptcies</td>
      <td>Number of public record bankruptcies</td>
    </tr>
  </tbody>
</table>


## Starter Codes


```python 

import pandas as pd
data_info = pd.read_csv('../DATA/lending_club_info.csv',index_col='LoanStatNew')
print(data_info.loc['revol_util']['Description'])
```
Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.

```python
def feat_info(col_name):
    print(data_info.loc[col_name]['Description'])
feat_info('mort_acc')
```
Number of mortgage accounts.


## Loading the data and other imports
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


%matplotlib inline

df = pd.read_csv('../DATA/lending_club_loan_two.csv')

df.info()

```
<img src= "https://user-images.githubusercontent.com/66487971/90667484-846d2900-e257-11ea-9400-f10b436df8e9.png" width = 300>


## EDA
```python
sns.countplot(x='loan_status',data=df)
```

<img src= "https://user-images.githubusercontent.com/66487971/90667904-e29a0c00-e257-11ea-98fa-4dfc000c9dcf.png" width = 400>

```python
plt.figure(figsize=(12,4))
sns.distplot(df['loan_amnt'],kde=False,bins=40)
plt.xlim(0,45000)
```

<img src= "https://user-images.githubusercontent.com/66487971/90668006-0b220600-e258-11ea-8fdf-8a919b6fd113.png" width = 600>

```python
df.corr()
```

<img src= "https://user-images.githubusercontent.com/66487971/90668140-4290b280-e258-11ea-9739-39710704df55.png" width = 1000>

```python
plt.figure(figsize=(12,7))
sns.heatmap(df.corr(),annot=True,cmap='viridis')
plt.ylim(10, 0)
```
<img src= "https://user-images.githubusercontent.com/66487971/90668375-ad41ee00-e258-11ea-9708-c1a17ef96503.png" width =860>

I have noticed almost perfect correlation with the "installment" feature.

```python
feat_info('installment')
```
The monthly payment owed by the borrower if the loan originates.

```python
feat_info('loan_amnt')
```
The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.

```python

sns.scatterplot(x='installment',y='loan_amnt',data=df,)

```
<img src= "https://user-images.githubusercontent.com/66487971/90669167-df078480-e259-11ea-953a-057b8af681cf.png" width =350>

```python
sns.boxplot(x='loan_status',y='loan_amnt',data=df)
```

<img src= "https://user-images.githubusercontent.com/66487971/90669258-02323400-e25a-11ea-8141-56b53951504d.png" width =350>

```python
df.groupby('loan_status')['loan_amnt'].describe()
```

<img src= "https://user-images.githubusercontent.com/66487971/90669312-2130c600-e25a-11ea-8826-6606d20ecb96.png" width = 700>

```python
sorted(df['grade'].unique())
```
['A', 'B', 'C', 'D', 'E', 'F', 'G']

```python

sorted(df['sub_grade'].unique())

```

['A1',
 'A2',
 'A3',
 'A4',
 'A5',
 'B1',
 'B2',
 'B3',
 'B4',
 'B5',
 'C1',
 'C2',
 'C3',
 'C4',
 'C5',
 'D1',
 'D2',
 'D3',
 'D4',
 'D5',
 'E1',
 'E2',
 'E3',
 'E4',
 'E5',
 'F1',
 'F2',
 'F3',
 'F4',
 'F5',
 'G1',
 'G2',
 'G3',
 'G4',
 'G5']
 
 ```python
 sns.countplot(x='grade',data=df,hue='loan_status')
 ```
 
 <img src= "https://user-images.githubusercontent.com/66487971/90669906-0ad73a00-e25b-11ea-89f4-2a838dc672b5.png" width = 400>
 
 ```python
 plt.figure(figsize=(12,4))
subgrade_order = sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade',data=df,order = subgrade_order,palette='coolwarm' )
```

<img src= "https://user-images.githubusercontent.com/66487971/90669988-322e0700-e25b-11ea-82d1-132d68f15592.png" width = 650>

```python
plt.figure(figsize=(12,4))
subgrade_order = sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade',data=df,order = subgrade_order,palette='coolwarm' ,hue='loan_status')
```

<img src= "https://user-images.githubusercontent.com/66487971/90670111-60abe200-e25b-11ea-9fa5-9bce3763dc96.png" width = 650>

```python
f_and_g = df[(df['grade']=='G') | (df['grade']=='F')]

plt.figure(figsize=(12,4))
subgrade_order = sorted(f_and_g['sub_grade'].unique())
sns.countplot(x='sub_grade',data=f_and_g,order = subgrade_order,hue='loan_status')
```

<img src= "https://user-images.githubusercontent.com/66487971/90670189-820cce00-e25b-11ea-8341-25f8c5a1c1b0.png" width = 650>

```python
df['loan_status'].unique()
```
array(['Fully Paid', 'Charged Off'], dtype=object)

```python

df['loan_repaid'] = df['loan_status'].map({'Fully Paid':1,'Charged Off':0})
df[['loan_repaid','loan_status']]
```

<img src= "https://user-images.githubusercontent.com/66487971/90670502-f5aedb00-e25b-11ea-805d-59d175a8024b.png" width = 200>

```python
df.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot(kind='bar')
```

<img src= "https://user-images.githubusercontent.com/66487971/90671086-dbc1c800-e25c-11ea-8791-6d7a9a366824.png" width = 500>

##  Data PreProcessing

```python
df.isnull().sum()
```
<img src= "https://user-images.githubusercontent.com/66487971/90671250-1b88af80-e25d-11ea-82a1-5d38bdd72e22.png = 200>
         
## Converting this Series to be in term of percentage of the total DataFrame

```python

100* df.isnull().sum()/len(df)

```

<img src= "https://user-images.githubusercontent.com/66487971/90671641-a9fd3100-e25d-11ea-90ab-b503253647d2.png" width = 200>

## I examine emp_title and emp_length to see whether it will be okay to drop them.

```python
feat_info('emp_title')
print('\n')
feat_info('emp_length')
```
The job title supplied by the Borrower when applying for the loan.


Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years. 

```python
df['emp_title'].nunique()
```
173105

```python
df['emp_title'].value_counts()
```

<img src= "https://user-images.githubusercontent.com/66487971/90673237-1bd67a00-e260-11ea-943a-ba9a96e0c2dc.png" width = 400>

**There are too many unique job titles to try to convert this to a dummy variable feature.**

```python

df = df.drop('emp_title',axis=1)

```
**Sorting the order of the values.**

```python
sorted(df['emp_length'].dropna().unique())
```

<img src= "https://user-images.githubusercontent.com/66487971/90673693-c484d980-e260-11ea-901e-b12a80eab34a.png" width = 100>

```python

emp_length_order = [ '< 1 year',
                      '1 year',
                     '2 years',
                     '3 years',
                     '4 years',
                     '5 years',
                     '6 years',
                     '7 years',
                     '8 years',
                     '9 years',
                     '10+ years']
```



```python

plt.figure(figsize=(12,4))

sns.countplot(x='emp_length',data=df,order=emp_length_order)
```

<img src= "https://user-images.githubusercontent.com/66487971/90674061-46750280-e261-11ea-87f4-5045d5e73e1e.png" width = 700>

```python
plt.figure(figsize=(12,4))
sns.countplot(x='emp_length',data=df,order=emp_length_order,hue='loan_status')
```

<img src= "https://user-images.githubusercontent.com/66487971/90674197-715f5680-e261-11ea-95ad-8c52bcbfb029.png" width = 700>


**Creating percentage of charge offs per category.**

```python

emp_co = df[df['loan_status']=="Charged Off"].groupby("emp_length").count()['loan_status']
emp_fp = df[df['loan_status']=="Fully Paid"].groupby("emp_length").count()['loan_status']
emp_len = emp_co/emp_fp
emp_len

```

<img src= "https://user-images.githubusercontent.com/66487971/90674454-d2872a00-e261-11ea-9609-a2b3b2242bf6.png" width = 300>

```python

emp_len.plot(kind='bar')
```

<img src= "https://user-images.githubusercontent.com/66487971/90674616-07937c80-e262-11ea-939c-80ec908c18cf.png" width = 400>


**Charge off rates are extremely similar across all employment lengths.**

```python
df = df.drop('emp_length',axis=1)
```

```python

df.isnull().sum()
```

<img src= "https://user-images.githubusercontent.com/66487971/90675030-9e603900-e262-11ea-982a-33924d58c94f.png" width = 200>

```python
df['purpose'].head(10)
```
<img src= "https://user-images.githubusercontent.com/66487971/90675356-04e55700-e263-11ea-88b4-c73a7a50e8ac.png" width = 250>

```python
df['title'].head(10)
```
<img src= "https://user-images.githubusercontent.com/66487971/90675413-1af31780-e263-11ea-8859-ed2278c218f7.png" width = 200>

**The title column is simply a string subcategory/description of the purpose column.**

```python
df = df.drop('title',axis=1)
```

```python
feat_info('mort_acc')
```
Number of mortgage accounts.


```python
df['mort_acc'].value_counts()
```

<img src= "https://user-images.githubusercontent.com/66487971/90675764-9359d880-e263-11ea-9f07-1f9b72e6f052.png" width = 250>

**Checking correlations to fill the empty values

```python
print("Correlation with the mort_acc column")
df.corr()['mort_acc'].sort_values()
```

<img src= "https://user-images.githubusercontent.com/66487971/90675913-cac88500-e263-11ea-897d-eaa594260c8c.png" width = 300>

**Looks like the total_acc feature correlates with the mort_acc**

```python

print("Mean of mort_acc column per total_acc")
df.groupby('total_acc').mean()['mort_acc']

```

<img src= "https://user-images.githubusercontent.com/66487971/90676336-85588780-e264-11ea-88cf-3eca049e35a9.png" width = 350>

**Filling in the missing mort_acc values based on their total_acc value.**

```python
total_acc_avg = df.groupby('total_acc').mean()['mort_acc']
```

```python

def fill_mort_acc(total_acc,mort_acc):
  
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc
        
```


```python
        
      df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)
      
      df.isnull().sum()
        
 ```

<img src= "https://user-images.githubusercontent.com/66487971/90676607-f4ce7700-e264-11ea-8643-24e1718f42c4.png" width = 200>


**revol_util and the pub_rec_bankruptcies have missing data points, but they account for less than 0.5% of the total data.**

```python
df = df.dropna()
```

## Categorical Variables and Dummy Variables

```python
df.select_dtypes(['object']).columns
```
Index(['term', 'grade', 'sub_grade', 'home_ownership', 'verification_status',
       'issue_d', 'loan_status', 'purpose', 'earliest_cr_line',
       'initial_list_status', 'application_type', 'address'],
      dtype='object')
      
```python
df['term'].value_counts()      
```
<img src= "https://user-images.githubusercontent.com/66487971/90677310-dae16400-e265-11ea-8610-a9376fc5dfeb.png" width = 200>

```python
df['term'] = df['term'].apply(lambda term: int(term[:3]))
```

```python
df = df.drop('grade',axis=1)
```
```python
subgrade_dummies = pd.get_dummies(df['sub_grade'],drop_first=True)
df = pd.concat([df.drop('sub_grade',axis=1),subgrade_dummies],axis=1)
df.columns
```
<img src= "https://user-images.githubusercontent.com/66487971/90677646-53482500-e266-11ea-8545-9f6b4d6acb29.png" width = 600>

```python
df.select_dtypes(['object']).columns
```

<img src= "https://user-images.githubusercontent.com/66487971/90677951-bc2f9d00-e266-11ea-855e-cbffbfcda536.png" width = 600>

```python
dummies = pd.get_dummies(df[['verification_status', 'application_type','initial_list_status','purpose' ]],drop_first=True)
df = df.drop(['verification_status', 'application_type','initial_list_status','purpose'],axis=1)
df = pd.concat([df,dummies],axis=1)
```

```python
df['home_ownership'].value_counts()
```


<img src= "https://user-images.githubusercontent.com/66487971/90678158-fdc04800-e266-11ea-9163-9c1cf4284e78.png" width = 150>


```python

df['home_ownership']=df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')

dummies = pd.get_dummies(df['home_ownership'],drop_first=True)
df = df.drop('home_ownership',axis=1)
df = pd.concat([df,dummies],axis=1)

```

```python
df['zip_code'] = df['address'].apply(lambda address:address[-5:])
```


```python

dummies = pd.get_dummies(df['zip_code'],drop_first=True)
df = df.drop(['zip_code','address'],axis=1)
df = pd.concat([df,dummies],axis=1)

```
**issue_d would be a data leakage.**


```python
df = df.drop('issue_d',axis=1)
```

**Extracting the year**

```python

df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda date:int(date[-4:]))
df = df.drop('earliest_cr_line',axis=1)

df.select_dtypes(['object']).columns

df = df.drop('loan_status',axis=1)

```

## Train Test Split

```python
from sklearn.model_selection import train_test_split

X = df.drop('loan_repaid',axis=1).values
y = df['loan_repaid'].values
```






































 
 

























































