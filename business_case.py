#!/usr/bin/env python
# coding: utf-8

# In[3]:



# Import librairies

import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize']= (10, 5)
sns.set_style('darkgrid')

rubikfont = {'fontname':'Rubik'}

# plt.title('title',**csfont)
# plt.xlabel('xlabel', **hfont)
# plt.show()


# In[4]:


# Import .csv file

df = pd.read_csv('C:/Users/digi/maxime_blanchard/business_case/Data - Business case - 30a18daf-d017-4344-b6f7-41e5909eb8c4.csv.csv')

df['date'] = pd.to_datetime(df['date'])

df['month'] = df['date'].dt.month_name()
df['week'] = df['date'].dt.isocalendar().week
df['day'] = df['date'].dt.day
df['day_name'] = df['date'].dt.strftime("%A")
df = df.set_index(['date'])
df = df.sort_index()

df['ecpm']=df['revenue']/df['impressions']*1000


# In[5]:


# Categorize months and days features

months = df['month'].unique()
days = df['day'].unique()
days_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df_2 = df.copy()
df['month'] = pd.Categorical(df_2['month'], months)
df['day'] = pd.Categorical(df_2['day'], days)
df['day_name'] = pd.Categorical(df_2['day_name'], days_name)


# In[7]:


# Quick overview of the dataset

# Infos about numeric features

# describe_df = df.describe()
# describe_df.reset_index(inplace=True)
# plt.factorplot(x="index", data=describe_num_df)  
# plt.show()


# Numeric features have a very large range of values (compare the median and the maximum value). There is no zero nor null value.

# In[ ]:





# # 1. Distribution of Numeric Features

# ## 1.1. Distribution of Each Numeric Feature

# In[8]:


fig, axes = plt.subplots(1, 2,figsize=(12,5))
fig.tight_layout(pad=5.0)
sns.histplot(data=df, x = 'ecpm', ax = axes[0], color = '#1d91c0', bins = 30).set(xlabel='eCPM Rate', ylabel = 'Count')
axes[0].set_title('eCPM Distribution', **rubikfont)
sns.histplot(data=df, x = 'ecpm', ax = axes[1], log_scale=True, color = '#1d91c0', bins = 30).set(xlabel='eCPM Log Rate', ylabel = 'Count')
axes[1].set_title('eCPM Distribution (log)', **rubikfont)
plt.show()


# **Observation** - eCPM distribution is non-linear and highly skewed. Potential outliers have to be defined with caution.

# In[9]:


fig, axes = plt.subplots(1, 2,figsize=(12,5))
fig.tight_layout(pad=5.0)
sns.histplot(data=df['impressions'], ax = axes[0], color = '#1d91c0', bins = 30).set(xlabel='Number of Impressions', ylabel = 'Count')
axes[0].set_title('Impressions Distribution')
sns.histplot(data=df['impressions'], ax = axes[1], log_scale=True, color = '#1d91c0', bins = 30).set(xlabel='Number of Impressions (log)', ylabel = 'Count')
axes[1].set_title('Impressions Distribution (log)')
plt.show()


# **Observation** - Impressions distribution is non-linear.

# In[10]:


fig, axes = plt.subplots(1, 2,figsize=(12,5))
fig.tight_layout(pad=5.0)
sns.histplot(data=df['revenue'], ax = axes[0], color = '#1d91c0', bins = 30).set(xlabel='Revenue Rate', ylabel = 'Count')
axes[0].set_title('Revenue Distribution')
sns.histplot(data=df['revenue'], ax = axes[1], log_scale=True, color = '#1d91c0', bins = 30).set(xlabel='Revenue Rate (log scale)', ylabel = 'Count')
axes[1].set_title('Revenue Distribution (log scale)')
plt.show()


# **Observation 1** - Revenue distribution is non-linear.
# 
# **Observation 2** - All numeric features follow a non-linear pattern. The mean can be misleading because it is sensible to outliers.

# ## 1.2. Outliers

# In[11]:


# ...


# **Temporary conclusion** - Now that the distribution of the numeric features has been described and its outliers studied, let's see how do they interact between each other and over time.

# # 2. Evolution of Numeric Features

# ## 2.1. Numeric Features Relationship

# In[12]:


#Normalizing numeric features

df['ecpm_max'] = df['ecpm']/df['ecpm'].max()*100
df['revenue_max'] = df['revenue']/df['revenue'].max()*100
df['impressions_max'] = df['impressions']/df['impressions'].max()*100


# In[13]:


linewidth = 3

fig, axes = plt.subplots(1, 1,figsize=(12,5))
sns.lineplot(data=df['revenue_max'], errorbar=None, linewidth = linewidth, color = '#0c2c84')
sns.lineplot(data=df['impressions_max'], errorbar=None, linewidth = linewidth, color = '#1d91c0')
sns.lineplot(data=df['ecpm_max'], errorbar=None, linewidth = linewidth, color = '#7fcdbb')
axes.legend(labels = ['Revenue', 'Impressions', 'eCPM'], title='Numeric Features')
axes.set(xlabel = 'Time', ylabel ='Normalized Ratio (%)')
axes.set_title('Normalized Evolution of eCPM, Impressions and Revenue Over Time')
plt.show()


# **Observation 1** - In June, impressions and revenue are strongly correlated. But from July to August, impressions increase, while revenue stays constant. It results in a lower eCPM.
# 
# **Observation 2** - Seasonality of impressions and revenue is observed. This point will be detailed in section 4.2.

# In[14]:


fig, ax = plt.subplots(1, 3, figsize =(10,6))

corr_jun = df.loc[df['month'] == 'June', ['impressions', 'revenue', 'ecpm']].corr()
corr_jul = df.loc[df['month'] == 'July', ['impressions', 'revenue', 'ecpm']].corr()
corr_aug = df.loc[df['month'] == 'August', ['impressions', 'revenue', 'ecpm']].corr()

sns.heatmap(corr_jun, ax = ax[0], annot = True, cmap = 'YlGnBu_r', linewidths=0.01, linecolor='white', square= True, cbar=False)
ax[0].set_title('June')
sns.heatmap(corr_jul, ax = ax[1], annot = True, cmap = 'YlGnBu_r', linewidths=0.01, linecolor='white', square= True, yticklabels='', cbar=False)
ax[1].set_title('July')
sns.heatmap(corr_aug, ax = ax[2], annot = True, cmap = 'YlGnBu_r', linewidths=0.01, linecolor='white', square= True, yticklabels='', cbar=False)
ax[2].set_title('August')
plt.suptitle('Correlation Between Numeric Features Over Time (Decrease)')
plt.legend('', frameon=False)
plt.show()


# **Observation** -  Correlation measurement between impressions and revenue shows a drop from 0.81 to 0.39 and consolidates observation 1 from the previous graph. The light green on the upper left of the heatmap turns darker.

# ## 2.2. Focus on eCPM Evolution

# In[15]:


ecpm_mean_jun = df.ecpm[df['month']=='June'].mean()
ecpm_mean_jul = df.ecpm[df['month']=='July'].mean()
ecpm_mean_aug = df.ecpm[df['month']=='August'].mean()

fig, axes = plt.subplots(1, 1,figsize=(12,5))
sns.lineplot(data=df[df['month']=='June'], x='day', y='ecpm', linewidth = linewidth, estimator='mean', errorbar=None, color = '#7fcdbb')
#mean_line = ax.plot(df['day'],ecpm_mean_jun, label='Mean', linestyle='--')
sns.lineplot(data=df[df['month']=='July'], x='day', y='ecpm', linewidth = linewidth, estimator='mean', errorbar=None, color = '#0c2c84')
sns.lineplot(data=df[df['month']=='August'], x='day', y='ecpm', linewidth = linewidth, estimator='mean', errorbar=None, color = '#1d91c0')
#mean_line_jun = ax.plot(data=ecpm_mean_jun)

axes.legend(labels = months, title='Month')
axes.set(xlabel = 'Day of the Month', ylabel ='eCPM (Mean)')
axes.set_title('eCPM Evolution by Month (Decrease)')
plt.xlim(left=0.5, right=31.5)
plt.show()

print("Mean of eCPM during June: " + str(ecpm_mean_jun))
print("Mean of eCPM during July: " + str(ecpm_mean_jul))
print("Mean of eCPM during August: " + str(ecpm_mean_aug))

print("Drop Evaluation from June to July: " + str((ecpm_mean_jul/ecpm_mean_jun)*100 - 100) + "%")


# **Observation** - From June to July, the overall eCPM drops of -11.06%.

# ## 2.3. Relationships Overview

# ![features-relationship.png](https://i.postimg.cc/DwfjSZbM/features-relationship.png)

# **Temporary conclusion** - The numeric features distribution and relations have been explored. Let's take a look at the categorical features in the next section and the multiple dependencies between and them and the numeric ones.

# # 3. Distribution of Categorical Features

# ## 3.1. Categorical Features Granularity

# [![g1749.png](https://i.postimg.cc/Xv65z2Rc/g1749.png)](https://postimg.cc/sMK1BJDv)

# **Observation** - This dataset is composed of categorical and numeric features. Categorical features are linked hierarchically. This hierarchy is named granularity. Website 1 and 2 share common adunits 1 and 9. Some sizes are specific to a type of device. The 'native' one appears only on website 2.

# ## 3.2. Analysis of Each Categorical Feature

# We will follow the top-down order of the granularity: 
# * website number; 
# * ad unit;
# * ad size;
# * type of device.

# ### 3.2.1. Website

# In[16]:


total_impressions = df['impressions'].sum()

fig, axes = plt.subplots(1,2,figsize=(12,5))
fig.tight_layout(pad=5.0)

sns.barplot(data = df, x='website', y ='impressions', estimator = 'sum' , ax = axes[0], hue = 'month', errorbar=None, palette='YlGnBu')
axes[0].set(xlabel = 'Website', ylabel ='Number of Impressions (x10M)')
axes[0].set_title('Total of Impressions by Website per Month')
axes[0].legend(title = 'Month', labels = months)

sns.barplot(data = df, x='website', y ='revenue' , estimator = 'sum', ax = axes[1], hue = 'month', errorbar=None, palette='YlGnBu')
axes[1].set(xlabel = 'Website', ylabel ='Total Revenue')
axes[1].set_title('Total Revenue by Website per Month')
axes[1].legend(title = 'Month', labels = months)
plt.show()


# **Observation 1** - Website 1 is insignificant in either feature and over time, thus, it can be ignored.
# 
# **Observation 2** - Website 2 increases in term of number of impressions, but sees a general decrease in term of revenue generated, so its eCPM is expected to drop.

# ### 3.2.2. Adunit

# In[17]:


fig, axes = plt.subplots(1,2,figsize=(12,5))
fig.tight_layout(pad=5.0)
sns.barplot(data = df, x='adunit', y ='impressions', estimator = 'sum', ax = axes[0], hue = 'month', errorbar=None, palette='YlGnBu')
sns.barplot(data = df, x='adunit', y ='revenue', estimator = 'sum', ax = axes[1], hue = 'month', errorbar=None, palette='YlGnBu')
axes[0].set(xlabel = 'Adunit', ylabel = 'Total of Impressions', title = 'Total of Impressions per Adunit (x10M)')
axes[1].set(xlabel = 'Adunit', ylabel = 'Total Revenue', title = 'Total Revenue per Adunit')
axes[0].legend(title='Month')
axes[1].legend(title='Month')
plt.show()


# **Observation 1** - Adunit 9 strongly increases in term of number of impressions, overtaking all the other adunits from July. But the revenue generated by it stays low compared to adunit 10's revenue.
# 
# **Observation 2** - Adunit 6 is insignificant either in term of impressions or revenue. It is due to the fact that it belongs exclusively to website 1 which generates very low impressions and revenue. To follow the order of granularity shows its importance here.
# 
# **Observation 3** - Adunit 10 decreases in both features, but its eCPM is expected to drop a bit too. Since it is the main source of revenue, the decrease in revenue of adunit 10 will be further examined in section 4.1.

# ### 3.2.3. Size

# In[18]:
    
df_select = df.loc[(df['adunit'] == 'adunit 9')|(df['adunit'] == 'adunit 10')].sort_values('adunit'
                                                                                           )

#%%

fig, axes = plt.subplots(1,2,figsize=(12,5))
fig.tight_layout(pad=7.0)
plt.xticks(rotation = 45)
sns.barplot(data = df_select, x=df_select['size'], y ='impressions', estimator = 'sum', hue = 'month', ax = axes[0], palette='YlGnBu', errorbar=None).set(
    title='Impressions per Ad Size per Month', xlabel='Ad Size', ylabel='Total Number of Impressions')
sns.barplot(data = df_select, x=df_select['size'], y ='revenue', estimator = 'sum', hue = 'month', ax = axes[1], palette='YlGnBu', errorbar=None).set(
    title='Total Revenue per Ad Size per Month', xlabel='Ad Size', ylabel='Total Revenue')
axes[0].legend(labels = months, title='Month')
axes[1].legend(labels = months, title='Month')
plt.setp( axes[0].xaxis.get_majorticklabels(), rotation=45 )
plt.show()


# **Observation 1** - 300x250 (from adunit 9) is the ad size generating the most impressions. It also strongly increases from June to July. Despite of this, 800x600 (from adunit 10) yields most of the revenue generated by the websites. Its eCPM is thus by far much greater than any other ad size.
# 
# **Observation 2** - 800x600 is constantly decreasing over time, either in terms of impression or in terms of revenue.

# ### 3.2.4. Type of Device

# In[19]:


fig, axes = plt.subplots(1, 2,figsize=(12,5))
fig.tight_layout(pad=5.0)
sns.barplot(data = df, x=df['device'], y ='impressions', estimator = 'sum', hue = 'month', ax = axes[0], errorbar=None, palette='YlGnBu').set(
    title='Total of Impressions by Device per Month', xlabel='Type of Device', ylabel='Total of Impressions (x10M)')
sns.barplot(data = df, x=df['device'], y ='revenue', estimator = 'sum', hue = 'month', ax = axes[1], errorbar=None, palette='YlGnBu').set(
    title='Total Revenue by Device per Month', xlabel='Type of Device', ylabel='Total Revenue')
axes[0].legend(labels = months, title='Month')
axes[1].legend(labels = months, title='Month')
plt.show()


# **Observation 1** - Desktop is the type of device generating the largest part of impressions and revenue. Mobile devices are almost insignificant.
# 
# **Observation 2** - The revenue does not follow the increase in term of impression. A drop in eCPM is thus expected.

# **Temporary conclusion** - A distinct disparity between groups of data emerged: for example, website 1 is insignificant compared to website 2 concerning the total revenue generated or total of impressions. By summing up all these groups, the main parameter influencing the whole eCPM evolution can be identified. This will be the subject of the next section.

# In[96]:


groupby = df.groupby(['website', 'adunit', 'size', 'device']).sum()


# In[72]:


df.groupby(['website', 'adunit', 'size', 'device', 'month']).sum().sort_values('impressions', ascending=False)


# In[95]:


df.groupby(['website', 'adunit', 'size', 'device']).sum().sort_values('revenue', ascending=False)


# In[87]:


total_impressions = df.impressions.sum()
total_revenue = df.revenue.sum()


# In[102]:


fig, axes = plt.subplots(1, 2,figsize=(12,5))

sns.barplot(data = df, x='size', y ='impressions', estimator='sum', ax = axes[0], hue = 'month', errorbar=None, palette='YlGnBu')
axes[0].set(xlabel = 'Website', ylabel ='impressions total')
axes[0].set_title('Total of Impressions by Website per Month')
axes[0].legend(title = 'Month', labels = months)
axes[0].set_ylim(0, total_impressions)


sns.barplot(data = df, x='size', y ='revenue', estimator='sum', ax = axes[1], hue = 'month', errorbar=None, palette='YlGnBu')
axes[1].set(xlabel = 'Website', ylabel ='revenue total')
axes[1].set_title('Total of Revenue by Website per Month')
axes[1].legend(title = 'Month', labels = months)
axes[1].set_ylim(0, total_revenue)
plt.show()


# In[ ]:


df.descrb


# # 4. Specific Observations

# ## 4.1. Targeted Analysis: the Main Parameter
# 
# **Definition of the main parameter** - The element *website2 - adunit 10 - 800x600 - desktop* yields more than 80% of the total revenue. Being the *main parameter*, it must be isolated and studied first and foremost.

# In[ ]:


df_main_parameter = df[(df['website'] == 'website 2')&(df['adunit'] == 'adunit 10')&(df['size'] == '800x600')&(df['device'] == 'desktop')]


# In[ ]:


fig, axes = plt.subplots(1, 1,figsize=(12,5))

sns.lineplot(data = df_main_parameter[df_main_parameter['month'] == 'June'], x='day', y='ecpm', errorbar=None, color = '#7fcdbb', linewidth=linewidth)
sns.lineplot(data = df_main_parameter[df_main_parameter['month'] == 'July'], x='day', y='ecpm', errorbar=None, color = '#1d91c0', linewidth=linewidth)
sns.lineplot(data = df_main_parameter[df_main_parameter['month'] == 'August'], x='day', y='ecpm', errorbar=None, color = '#0c2c84', linewidth=linewidth)

axes.legend(labels = months, title='Month')
axes.set(xlabel = 'Day of the Month', ylabel ='eCPM')
axes.set_title('eCPM Evolution of the Main Parameter by Month (Decrease)')
plt.xlim(left=0.5, right=31.5)
plt.show()

mean_jun = df_main_parameter.ecpm[df_main_parameter['month'] == 'June'].mean()
mean_jul = df_main_parameter.ecpm[df_main_parameter['month'] == 'July'].mean()
mean_aug = df_main_parameter.ecpm[df_main_parameter['month'] == 'August'].mean()

print("Mean of the main parameter during June: " + str(mean_jun))
print("Mean of the main parameter during July: " + str(mean_jul))
print("Mean of the main parameter during August: " + str(mean_aug))

print("Drop Evaluation: " + str((mean_aug/mean_jun)*100-100) +"%")


# **Observation** - The main parameter describes a constant drop of eCPM from 22.4 during June to 19.7 in August. It represents a 11.7% decrease.

# ## 4.2. Seasonality of Impressions

# In[ ]:


fig, axes = plt.subplots(1, 1, figsize=(5,7))

sns.barplot(data = df_main_parameter, x = 'day_name', y = 'impressions', estimator = 'mean', errorbar=None, palette='YlGnBu').set(
    title='Total of Impressions by Device per Month')

axes.set(xlabel = 'Day of the Week', ylabel ='Impressions (Mean)')
plt.xticks(rotation = 45)
axes.set_title('Seasonality of Impressions')
plt.show()


# **Observation** - The number of impressions  of the websites is affected by seasonality: saturday and sunday display a significant difference regarding other days of the week. The regular drop observed in the graph of section 2.2 are identified as the weekend (saturday and sunday).

# # 5. Conclusion
# 
# The eCPM decrease following the end of June is the result of a high increase of impressions due to adunit 9 (see sections 3.2. and 3.3.) which has a low revenue yield. Coupled with the decrease of the role played by the main parameter in the overall revenue yielding, the eCPM had to drop.

#%%

gb_ws = df.groupby('size').sum()
pc_ws_imp = gb_ws['impressions'] / df['impressions'].sum() *100
pc_ws_rev = gb_ws['revenue'] / df['revenue'].sum() *100

#%%

loc_factor = df.loc[(df['website']=='website 2') & (df['adunit'] == 'adunit 10') & 
       (df['size'] == '800x600') & (df['device'] == 'desktop') & (df['month'] == 'June')].sum()

#%%

pc_loc_imp = loc_factor['impressions'] / df.loc((df['impressions'])&(df['month'] == 'June')).sum() *100
pc_loc_rev = loc_factor['revenue'] / df['revenue'].sum() *100

#%%

loc_factor_jun = df.loc[(df['website']=='website 2') & (df['adunit'] == 'adunit 10') & 
       (df['size'] == '800x600') & (df['device'] == 'desktop') & (df['month'] == 'June')]

loc_factor_jul = df.loc[(df['website']=='website 2') & (df['adunit'] == 'adunit 10') & 
       (df['size'] == '800x600') & (df['device'] == 'desktop') & (df['month'] == 'July')]

loc_factor_aug = df.loc[(df['website']=='website 2') & (df['adunit'] == 'adunit 10') & 
       (df['size'] == '800x600') & (df['device'] == 'desktop') & (df['month'] == 'August')]


df_jun = df.loc[df['month'] == 'June']

df_jul = df.loc[df['month'] == 'July']

df_aug = df.loc[df['month'] == 'August']

pc_fac_rev_jun = loc_factor_jun.revenue.sum() / df_jun.revenue.sum() *100
pc_fac_rev_jul = loc_factor_jul.revenue.sum() / df_jul.revenue.sum() *100
pc_fac_rev_aug = loc_factor_aug.revenue.sum() / df_aug.revenue.sum() *100

pc_fac_rev_ls = [pc_fac_rev_jun, pc_fac_rev_jul, pc_fac_rev_aug]

pc_fac_imp_jun = loc_factor_jun.impressions.sum() / df_jun.impressions.sum() *100
pc_fac_imp_jul = loc_factor_jul.impressions.sum() / df_jul.impressions.sum() *100
pc_fac_imp_aug = loc_factor_aug.impressions.sum() / df_aug.impressions.sum() *100

#%%     

dic = {'impressions':{'june':pc_fac_imp_jun, 
                      'july':pc_fac_imp_jul,
                      'august':pc_fac_imp_aug},
       'revenue':{'june':pc_fac_rev_jun,
                  'july':pc_fac_rev_jul,
                  'august':pc_fac_rev_aug}}

resume_df = pd.DataFrame.from_dict(dic, orient='index')

#%%

df_mean = df.groupby(['website', 'adunit', 'size', 'device']).mean()

#%%%%%

loc_detr_jun = df.loc[(df['website']=='website 2') & (df['adunit'] == 'adunit 9') & 
       (df['size'] == '300x250') & (df['device'] == 'desktop') & (df['month'] == 'June')]

loc_detr_jul = df.loc[(df['website']=='website 2') & (df['adunit'] == 'adunit 9') & 
       (df['size'] == '300x250') & (df['device'] == 'desktop') & (df['month'] == 'July')]

loc_detr_aug = df.loc[(df['website']=='website 2') & (df['adunit'] == 'adunit 9') & 
       (df['size'] == '300x250') & (df['device'] == 'desktop') & (df['month'] == 'August')]


#%%
df_date = df.reset_index()
#%%

df_date.loc[(df_date['date'] < '2022-07-15')].ecpm.mean()

#%%

df_week = df.groupby(['week', 'month']).mean().sort_values('ecpm', ascending = False)

df_week =df_week.sort_values('week').reset_index()
#%% Plotting
fig, ax = plt.subplots(1,1, figsize=(12,5))
sns.barplot(data = df_week, x = 'week', y = 'ecpm', errorbar=None, palette='YlGnBu', hue = 'month').set(
    title='eCPM Evolution by Week', xlabel='Week of the Year N°', ylabel='eCPM (€/1000 imp.)')
ax.legend(labels = months, title='Month')
#%%
df_week.sort_values('ecpm', ascending = False)

#%%

loc_benef = df.loc[(df['website']=='website 2') & (df['adunit'] == 'adunit 10') & 
       (df['size'] == '800x600') & (df['device'] == 'desktop')]

loc_detr = df.loc[(df['website']=='website 2') & (df['adunit'] == 'adunit 9') & 
       (df['size'] == '300x250') & (df['device'] == 'desktop')]

#%%
fig, ax = plt.subplots(1,2, figsize=(12,5))
fig.tight_layout(pad=5.0)

sns.barplot(data = loc_benef, x = 'month',  y = 'impressions', width = 0.55, estimator = 'sum', ax = ax[0],errorbar = None, palette='YlGnBu')
sns.barplot(data = loc_benef, x = 'month',  y = 'revenue', width = 0.55,estimator = 'sum', ax = ax[1], errorbar = None, palette='YlGnBu')
ax[0].set_title('Beneficial Factor\'s Impressions Evolution')
ax[0].set(xlabel='Month', ylabel='Number of Impressions (x10M)')
ax[1].set_title('Beneficial Factor\'s Revenue Evolution')
ax[1].set(xlabel='Month', ylabel='Total Revenue')
#%%

fig, ax = plt.subplots(1,2, figsize=(12,5))
fig.tight_layout(pad=5.0)

sns.barplot(data = loc_detr, x = 'month', y = 'impressions', width = 0.55, errorbar = None, estimator = 'sum', ax = ax[0], palette='YlGnBu')
sns.barplot(data = loc_detr, x = 'month',  y = 'revenue',estimator = 'sum', width = 0.55, palette='YlGnBu', errorbar = None, ax = ax[1])
ax[0].set_title('Detrimental Factor\'s Impressions Evolution',)
ax[0].set(xlabel='Month', ylabel='Number of Impressions (x10M)')
ax[1].set_title('Detrimental Factor\'s Revenue Evolution')
ax[1].set(xlabel='Month', ylabel='Total Revenue')

#%% STACKED BAR CHARTS
total = df[['impressions', 'revenue', 'month']]
total_june = df[['impressions', 'revenue', 'month']].loc[df['month'] == 'June']
total_july = df[['impressions', 'revenue', 'month']].loc[df['month'] == 'July']
total_august = df[['impressions', 'revenue', 'month']].loc[df['month'] == 'August']

total_june.impressions.sum()

sns.barplot(x="month",  y="impressions", estimator = 'sum', data=total, color='darkblue')
#%%

df.plot(kind = 'bar', stacked = True)

#%%

df['benef'] = 'Other Factors'
df['detr'] = 'Other Factors'

#%%

df.loc[((df['website']=='website 2') & (df['adunit'] == 'adunit 10') & 
       (df['size'] == '800x600') & (df['device'] == 'desktop')), 'benef'] = 'Beneficial Factor'

#%%

df.loc[((df['website']=='website 2') & (df['adunit'] == 'adunit 9') & 
       (df['size'] == '300x250') & (df['device'] == 'desktop')), 'detr'] = 'Detrimental Factor'

#%%

cols_benef = ['red' if x == 'Beneficial factor' else 'grey' for x in df.benef]



fig, ax = plt.subplots(1,2, figsize=(12,5))
fig.tight_layout(pad=5.0)
sns.barplot(data = df, x = 'month', y='impressions', hue = 'benef', ax=ax[0], estimator = 'sum', errorbar = None, palette='YlGnBu_r')
sns.barplot(data = df, x = 'month', y='revenue', hue = 'benef', ax=ax[1], estimator = 'sum', errorbar = None, palette='YlGnBu_r')
ax[0].set_title('Role of the Beneficial Factor (impressions)')
ax[1].set_title('Role of the Beneficial Factor (revenue)')
ax[0].legend(title='Factors')
ax[1].legend(title='Factors')
ax[0].set(xlabel = 'Month', ylabel = 'Number of Impressions')
ax[1].set(xlabel = 'Month', ylabel = 'Total Revenue')


#%%
fig, ax = plt.subplots(1,2, figsize=(12,5))
fig.tight_layout(pad=5.0)
sns.barplot(data = df, x = 'month', y='impressions', hue = 'detr', ax=ax[0], estimator = 'sum', errorbar = None, palette='YlGnBu_r')
sns.barplot(data = df, x = 'month', y='revenue', hue = 'detr', ax=ax[1], estimator = 'sum', errorbar = None, palette='YlGnBu_r')
ax[0].set_title('Role of the Detrimental Factor (impressions)')
ax[1].set_title('Role of the Detrimental Factor (revenue)')
ax[0].legend(labels = df.detr.unique(), title='Factors')
ax[1].legend(title='Factors')
ax[0].set(xlabel = 'Month', ylabel = 'Number of Impressions')
ax[1].set(xlabel = 'Month', ylabel = 'Total Revenue')

#%%

df['impressions_percent'] = df.impressions/df.impressions.sum()*100
df['revenue_percent'] = df.revenue/df.revenue.sum()*100

#%%

fig, ax = plt.subplots(1,2, figsize=(12,5))
fig.tight_layout(pad=5.0)
sns.barplot(data = df, x = 'month', y='impressions', hue = 'benef', ax=ax[0], estimator = 'sum', errorbar = None, palette='YlGnBu_r')
sns.barplot(data = df, x = 'month', y='revenue', hue = 'benef', ax=ax[1], estimator = 'sum', errorbar = None, palette='YlGnBu_r')
ax[0].set_title('Role of the Beneficial Factor (impressions)')
ax[1].set_title('Role of the Beneficial Factor (revenue)')
ax[0].legend(title='Factors')
ax[1].legend(title='Factors')
ax[0].set(xlabel = 'Month', ylabel = 'Number of Impressions')
ax[1].set(xlabel = 'Month', ylabel = 'Total Revenue')


#%%
fig, ax = plt.subplots(1,2, figsize=(12,5))
fig.tight_layout(pad=5.0)
sns.barplot(data = df, x = 'month', y='impressions', hue = 'detr', ax=ax[0], estimator = 'sum', errorbar = None, palette='YlGnBu_r')
sns.barplot(data = df, x = 'month', y='revenue', hue = 'detr', ax=ax[1], estimator = 'sum', errorbar = None, palette='YlGnBu_r')
ax[0].set_title('Role of the Detrimental Factor (impressions)')
ax[1].set_title('Role of the Detrimental Factor (revenue)')
ax[0].legend(labels = df.detr.unique(), title='Factors')
ax[1].legend(title='Factors')
ax[0].set(xlabel = 'Month', ylabel = 'Number of Impressions')
ax[1].set(xlabel = 'Month', ylabel = 'Total Revenue')

#%%

cross_tab_prop = pd.crosstab(index=df['impressions'],
                             columns=df['month'],
                             normalize="index")
cross_tab_prop

#%%

cross_tab = pd.crosstab(index=df['adunit'],
                        columns=df['impressions'])
cross_tab
#%%

plt.bar(month , color='r')
plt.bar(month, y2, bottom=y1, color='b')
plt.show()