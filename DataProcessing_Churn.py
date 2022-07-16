# Artificial Neural Network
# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import cassandra

# export ASTRA_DB_ID=4fb6aec8-e9ff-44db-8563-d3d3e9c7541a
# export ASTRA_DB_REGION=us-east1
# export ASTRA_DB_KEYSPACE=datasciencedb
# export ASTRA_DB_APPLICATION_TOKEN=AstraCS:YcRucwHCMLJnhBzuNRBQDFpM:789d93329cc9ddcfe79c1123fd6860ff6c3c2df7c7dee15cd7d927580f5940a9
#------------------------------------------------
# Importing the dataset
churn_df = pd.read_csv('D:\\Learn & Projects\\DeepLearning\\Dataset\\Churn_ANN.csv')

# Find the missing value or blank
print(churn_df.isnull().sum())
print(churn_df.isna().sum())
print(np.where(churn_df.applymap(lambda x: x == '')))
# No null values or NA values or blank values

# Remove the first 3 columns as they are non cor-related to dependent variable
df_eda = churn_df.iloc[:,3:14]

# Pie Chart to show the response variable distribution in dataset
values = df_eda.Exited.value_counts()
labels = ['Not Exited', 'Exited']

fig, ax = plt.subplots(figsize = (4, 3), dpi = 100)
explode = (0, 0.09)

patches, texts, autotexts = ax.pie(values, labels = labels, autopct = '%1.2f%%', shadow = True,
                                   startangle = 90, explode = explode)

plt.setp(texts, color = 'grey')
plt.setp(autotexts, size = 8, color = 'white')
autotexts[1].set_color('black')
plt.show()

# Visualizing categorical variables
colnames = list(['Geography',
 'Gender', 
 'Tenure',
 'NumOfProducts',
 'HasCrCard',
 'IsActiveMember'])


fig, ax = plt.subplots(3, 2, figsize = (25, 25))
col_index=np.arange(0, len(colnames), 1)
index =0
for i in range(0,int(len(colnames)/2)):
   
    if index > len(colnames):
        break
    else:
        
        for j in range (0,2):
            if j>1: 
                break
            else:
                plots =sns.countplot(colnames[col_index[index]], hue = 'Exited', data = df_eda, ax = ax[i][j])
                for bar in plots.patches:
                            plots.annotate(format(bar.get_height(), '.0f'),
                                           (bar.get_x() + bar.get_width() / 2, 
                                            bar.get_height()+5), ha='center', 
                                           va='center', size=12, 
                                           xytext=(0, 8),textcoords='offset points')
                index = index+1

plt.tight_layout()
plt.show()

# Obervations:

# Germany has lowest number of customers recorded in dataset and highest number 
#  of people who have exited may be due less availbility.
# Spain has lowest number of exited people
# France has highest non-exited people and majority of the people are from here.
# Germany has lowest number of people who have not exited
# Female have highest number of exiting
# Male have highest number of non-exiting
# People with tenure 1 year have exited highest
# People with 1 product have exited highest and people with lowest exited has 4 products
# People who have credit card have exited highest
# People who are active member have exited highest


geography_df_exited = df_eda[['Geography','Gender','Exited']]
geography_df_exited = geography_df_exited.value_counts().to_frame('Counts').reset_index()
#geography_df_exited
a = sns.catplot(x='Geography',y='Counts',data=geography_df_exited,hue='Exited',col='Gender',kind="bar",
                height=9, aspect=.7)    
# Fianlly showing the plot
plt.tight_layout()


# Visualizing the continous columns
continous_colnames = ['CreditScore','Age','Balance','EstimatedSalary']
fig, ax = plt.subplots(2, 2, figsize = (15, 15))
con_index=np.arange(0, len(continous_colnames), 1)
index=0
for i in range(0,int(len(continous_colnames)/2)):
    if index<len(continous_colnames):
        for j in range(0,2):
            if j<2:
                p = sns.boxplot(x="Exited", y=continous_colnames[con_index[index]],data=df_eda, linewidth=2.5,ax = ax[i][j])
                #p = sns.swarmplot(x="Exited", y=continous_colnames[con_index[index]],data=df_eda, linewidth=2.5,ax = ax[i][j],color=".25")                
                p=sns.stripplot(x="Exited", y=continous_colnames[con_index[index]],data=df_eda,ax = ax[i][j])
                index = index+1
            else: break
            
                
    else:
        break
        
plt.tight_layout()
plt.show()

# Observations:

# 1. Credit Score distribution of exited and non exited people are almost 
#    same so this means that credit score is not the factor for exiting
# 2. People with age approximately 35 and above are exiting more.
#     May be they are using less calls hence switching to cheaper carrier
# 3. Poeple who are having bank balance >50K are more prone to exiting
# 4. There are people with 0 Bank Balance , may be its a student (non-salaried)
#     account and they have not exited yet.
# 5. Estimated salary distribution has less/minimal impact on people exiting.


ax = sns.stripplot(x="Geography", y="Balance", hue="Exited", data=df_eda)
plt.show()

# Observation:
# People in France and Spain are having people with 0 Bank Balance 
# but Germany doesn't have such accounts may be more 
# salaried people in Germany and more students in Spain and France.

plt.figure(figsize=(30,10))
ax = sns.stripplot(x="Age", y="Balance", hue="Exited", data=df_eda)

# Older people with Bank Balance greater than 50K(approximately) are exiting more 

# Correlation Heat Map
# heatmap 

plt.figure(figsize = (20, 12))
sns.heatmap(df_eda.corr(), linewidths = 1, annot = True, fmt = ".2f")
plt.show()



