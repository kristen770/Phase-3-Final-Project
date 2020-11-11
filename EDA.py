#!/usr/bin/env python
# coding: utf-8

# # Final Project Submission
# 
# Please fill out:
# * Student name: Kristen Davis
# * Student pace: Full Time
# * Scheduled project review date/time: 11/16 @ 11:30
# * Instructor name: Rafael Carrasco

# ## Set Up & Data Initialization 

# In[1]:


#Libraries 
import pandas as pd    
import numpy as np

import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px 
import plotly.graph_objects as go 
from plotly.subplots import make_subplots  
get_ipython().run_line_magic('matplotlib', 'inline')


import plotly.io as pio
pio.renderers.default='notebook' 

import pickle


# In[2]:


#Functions
def column_inspect(colname): 
    print("Number of NaNs:", df[[colname]].isna().sum())
    print("Unique Values:", df[colname].unique()) 
    print("Number of Unique Values:", df[colname].nunique())  
    
    fig = px.histogram(df, x=colname)  
    fig.update_layout(barmode='group')
    fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)', marker_line_width=1.5, opacity=0.6) 
    fig.update_layout(title_text=colname)
    fig.show() 
    
    fig = px.histogram(df, x=colname, color='status_group')  
    fig.update_layout(barmode='group')
    fig.update_traces(marker_line_color='rgb(8,48,107)', marker_line_width=1.5, opacity=0.6) 
    fig.show()
    
def compare_columns(df, x, color):
    fig = px.histogram(df, x=x, color=color) 
    fig.update_layout(barmode='group') 
    fig.update_traces(marker_line_color='rgb(8,48,107)', marker_line_width=1.5, opacity=0.6) 
    fig.show()


# ### Load Data Sets

# In[3]:


#Data for this contest downloaded from datadriven.org
test_set_values = pd.read_csv('/Users/kristen/Downloads/702ddfc5-68cd-4d1d-a0de-f5f566f76d91.csv')  
training_set_labels = pd.read_csv('/Users/kristen/Downloads/0bf8bc6e-30d0-4c50-956a-603fc693d966.csv')   
training_set_values = pd.read_csv('/Users/kristen/Downloads/4910797b-ee55-40a7-8668-10efd5c1b960.csv') 


# In[4]:


#Merge training set
original_df = pd.merge(training_set_labels, training_set_values, on="id")  


# In[5]:


df = original_df


# In[6]:


is_NaN = df.isnull()
row_has_NaN = is_NaN.any(axis=1)
rows_with_NaN = df[row_has_NaN] 
print(rows_with_NaN)


# Structure:
# * The data has 59,400 unique enteries with 41 features(columns)  
# 
# Nan Values: 
# * 3,635 values are missing in the funder & installer category (6% of data missing) 
# * 371 values are missing from the subvillage category (>1% of data missing) 
# * 3,334 values are missing from the public meeting category (5% of data missing)  
# * 3,877 values are missing from the scheme management category (6.5% of data missing)
# * 28,166 values are missing from the scheme name category (47% of data missing) 
# * 3,056 values are missing from the permit category (5% of data missing) 

# ### Feature Correclation in Raw Data

# In[7]:


corrMatrix = df.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()


# In[8]:


pd.plotting.scatter_matrix(df, alpha=0.2) 
plt.show()


# In[9]:


df.apply(lambda x: x.factorize()[0]).corr()


# ### Column Types

# In[10]:


df.columns


# In[11]:


df.info()


# In[12]:


#itrative cell to look a unique values to identify type 
df['permit'].unique()


# * <b>Reference</b> - 'id', 'date_recorded', 'recorded_by'
# 
# 
# * <b>Categorical</b> - 'status_group', 'funder', 'installer', 'wpt_name', 'basin', 'subvillage', 'region', 'region_code', 'district_code', 'lga', 'ward', 'public_meeting', 'scheme_management', 'scheme_name', 'permit', 'construction_year', 'extraction_type', 'extraction_type_group', 'extraction_type_class', 'management', 'management_group', 'payment', 'payment_type', 'water_quality', 'quality_group', 'quantity', 'quantity_group', 'source', 'source_type', 'source_class', 'waterpoint_type', 'waterpoint_type_group'
# 
# 
# * <b>Continous</b> - 'amount_tsh', 'gps_height', 'longitude', 'latitude', 'num_private', 'population'

# ## Investigate Each Reference Column 

# ### id  
# 

# In[13]:


column_inspect('id')


# 59400 unique identifing ids correlated to a unique water pump. 

# ### date_recorded

# In[14]:


column_inspect('date_recorded')


# * Date data was recorded for each pump.  
# * Non NaN values  
# * The majority of the obeservations after 2010 with the largest number of data (572) recorded in March of 2011

# ### recorded_by

# In[15]:


column_inspect('recorded_by')


# * All values/ observations recorded by GeoData Consultants Ltd
# * No NaN values

# ## Investigate Each Categorical Column

# ### status_group

# In[16]:


column_inspect('status_group')


# In[17]:


labels = ['Functional', 'Non Functional', 'Functional Needs Repair']
values = [32259, 22824, 4317]

fig = go.Figure(data=[go.Pie(labels=labels, values=values)]) 
fig.update_traces(textposition='inside', textinfo='percent+label', title='Water Pump Status')
fig.show()


# In[18]:


#divide data by status_group
functional = df.loc[df['status_group'] == 'functional'] 
non_functional = df.loc[df['status_group'] == 'non functional'] 
functional_needs_repair = df.loc[df['status_group'] == 'functional needs repair']  


# * These are the categories that ultimately the model is attempting to sort into. Functional, non functional, and funtional needs repair 
# * 54.3% Functional Pumps/ 38.4% Non Functional/ 7.27% Functional Needs Repair 
# 
# 61.57% of the pumps in the data set are functional

# ### funder

# In[19]:


column_inspect('funder')


# In[20]:


funders_df = df.groupby('funder').count().reset_index()
funders_df_100 = funders_df.loc[(funders_df['id'] > 300)]


# In[21]:


fig = go.Figure(data=[go.Bar(x=funders_df_100['funder'], y=funders_df_100['id'])])
# Customize aspect
fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5, opacity=0.6)
fig.update_layout(title_text='Water Pump Funders')
fig.show()


# * funder represent the person/ entity who paid for the pump
# * 3635 values are missing funder 
# * Many of these enteries appear to be duplicates ie "Zao Water Spring/ Zao Water Spring X/ Zao" for example 
# * Rows that have a 0 listed as a funder also have a 0 listed as an installer 
# * 15% of all pumps were funded by the Government of Tanzania (9,084)
# * 1897 unique funders 
# * The average funder only made 29 pumps
# * At the 75% of the quantile only 7 pumps 
# * If you consider additional governemt services (District Council & Water) the government funderd 10,510 or 18%

# ### installer

# In[22]:


column_inspect('installer')


# In[23]:


installer_df = df.groupby('installer').count().reset_index()  
#installer_df['installer'].sort_values(ascending=False)
installer_df_100 = installer_df.loc[(installer_df['id'] > 150)] 
installer_df_100


# In[24]:


#Fix Obvious Value Typo
#df['installer'] = df['installer'].replace(['Commu'], 'Community') 
#df['installer'] = df['installer'].replace(['District council'], 'District Council') 
#df['installer'] = df['installer'].replace(['Gover'], 'Government') 
#df['installer'] = df['installer'].replace(['Hesawa'], 'HESAWA') 
#df['installer'] = df['installer'].replace(['World vision'], 'World Vision') 
df['installer'] = df['installer'].replace(['Gove'], 'Government')


# In[25]:


fig = go.Figure(data=[go.Bar(x=installer_df_100['installer'], y=installer_df_100['id'])])
# Customize aspect
fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5, opacity=0.6)
fig.update_layout(title_text='Water Pump Funders')
fig.show()


# * installer is the person or entity that installed the well
# * 3635 missing values  
# * Rows that have a 0 listed as a funder also have a 0 listed as an installer  
# * DWE = District Water Engineer  
# * If you agragate all the grovernement agencies and do a little reserach on the other organizations  
# * In May 2009 Water Supply and Sanitation Act Nr. 12 allowed the government agencies involved in water to act as commerical entities. Much like the American Water Works a publicly traded company that provides a 2% yield.

# ### wpt_name

# In[26]:


df['wpt_name'].nunique()


# In[27]:


wpt_names = df['wpt_name'].tolist() 
matches = [match for match in wpt_names if "School" in match] 
len(matches)


# * wpt = The Name of the Water Point
# * 63% of this column are unique values (37400)
# * 1,194 of these names contains the word School/ school in it so at least 3% of the pumps are located at schools

# ### basin

# In[28]:


column_inspect('basin')


# In[29]:


lake_nyasa = df.loc[df['basin'] == 'Lake Nyasa'] 
lake_victoria = df.loc[df['basin'] == 'Lake Victoria'] 
pangani = df.loc[df['basin'] == 'Pangani']
ruvuma = df.loc[df['basin'] == 'Ruvuma / Southern Coast'] 
internal = df.loc[df['basin'] == 'Internal'] 
tanganyika = df.loc[df['basin'] == 'Lake Tanganyika']  
wami = df.loc[df['basin'] == 'Wami / Ruvu'] 
rufiji = df.loc[df['basin'] == 'Rufiji'] 
lake_rukwa = df.loc[df['basin'] == 'Lake Rukwa'] 


# In[30]:


status = ['Functional', 'Functional Needs Repair', 'Non Functional']
fig = go.Figure(data=[
    go.Bar(name='Lake Nyasa', x=status, y=[3324, 250, 1511]),
    go.Bar(name='Lake Victoria', x=status, y=[5100, 989, 4159]),
    go.Bar(name='Pangani', x=status, y=[5372, 477, 3091]), 
    go.Bar(name='Ruvuma / Southern Coast', x=status, y=[1670, 326, 2497]), 
    go.Bar(name='Internal', x=status, y=[14482, 557, 2746]), 
    go.Bar(name='Lake Tanganyika', x=status, y=[3107, 742, 2583]), 
    go.Bar(name='Wami / Ruvu', x=status, y=[3136, 269, 2582]), 
    go.Bar(name='Rufiji', x=status, y=[5098, 437, 2471]), 
    go.Bar(name='Lake Rukwas', x=status, y=[1000, 270, 1184])
])
# Change the bar mode
fig.update_layout(barmode='group') 
fig.update_traces(marker_line_color='rgb(8,48,107)', marker_line_width=1.5, opacity=0.6) 
fig.update_layout(title_text='Water Pump Status By Basin')
fig.show()


# In[31]:


basins_status = df.groupby(['basin','status_group', 'payment_type']).count().reset_index()
basins_status.loc[(basins_status['payment_type'] == 'never pay') & (basins_status['status_group'] == 'functional')]


# In[32]:


compare_columns(df = df, x='basin', color='payment_type')


# In[33]:


compare_columns(df, x='basin', color='source')  


# In[34]:


compare_columns(df, x='basin', color='quantity')  


# * Geographical water basin 
# * The most pumps correlate to the Lake Victoria Basin / the least to Lake Rukwa  
# * The majority of pumps in each basin are never pay  
# * The largest number of never pay/ functioning pumps is in Lake Victoria (3061) 
# * The vast majority of water in each basin is of good or 'soft' quality 
# * Accross all basins springs are the predomenat water source  
# * Pangani is the water basin with the most insufficient water supply

# ### subvillage 

# In[35]:


column_inspect('subvillage')


# * Subvillages = Geographical Location Data / Subvillage pump is located at 
# * 371 Subvillages missing  
# * 19287 Unique Subvillage listings  
# * Madukani 508 / Shuleni 506 / Majengo 502 are the top 3 subvillages with the most water pumps 

# ### region

# In[36]:


column_inspect('region')


# In[37]:


compare_columns(df, x='region', color='payment')  


# In[38]:


compare_columns(df, x='region', color='quantity')  


# * Region - Broader Geographical Location Data  
# * 21 Unique Regious 
# * No Nan Values 
# * The regin with the most water pumps is Iringa 
# * The Iringa region has the highest percentage of functioning to non functioning water pumps 
# * The Mbeya has the largest percentage of non function pumps compared to functioning  
# * Iringa, the region with the highest number of functioning pumps also has more pay based pumns than non pay pumps

# ### region_code

# In[39]:


column_inspect('region_code')


# In[40]:


basin_code = df.groupby(['basin','region_code']).count()
basin_code


# In[41]:


funtion_code = df.groupby(['status_group', 'region_code']).count().reset_index()
funtion_code


# * Region code - Coded Geographical Data 
# * No NaN values
# * There is overlap in many of the basins/ region code 
# * I could not find any map that correlated to region coded listed  
# * There is a large clustering of region codes below 20 - could indicate those are the more dense areas

# ### district_code

# In[42]:


column_inspect('district_code')


# In[43]:


district_code = df.groupby(['status_group', 'district_code']).count().reset_index()
district_code


# * Distric_code - Coded Geographical Data 
# * Non NaN values 
# * This grouping is smaller than 'region' 

# ### lga

# In[44]:


column_inspect('lga')


# In[45]:


lga = df.groupby('lga').count().reset_index()


# * LGA - Local Governmental Authority accociated with Geographical Location 
# * 125 unique Local Authorities 

# ### ward

# In[46]:


column_inspect('ward')


# In[47]:


wards = df.groupby(['basin','lga', 'ward']).count().reset_index()
wards


# In[48]:


fig = px.sunburst(wards, path=['basin', 'lga', 'ward'], values='id')
fig.show()


# In[49]:


ward_lga = df.groupby(['lga', 'ward']).count()


# * Ward - Geographical Data, within LGA 
# * No NaN

# ### payment

# In[50]:


column_inspect('payment')


# In[51]:


compare_columns(df=functional, x='basin', color='payment')  


# In[52]:


compare_columns(df=df, x='payment', color='water_quality')  


# In[53]:


compare_columns(df=df, x='payment', color='quantity')  


# * The payment types: 'pay annually', 'never pay', 'pay per bucket', 'unknown', 'pay when scheme fails', 'other', 'pay monthly'

# ### payment_type

# In[54]:


column_inspect('payment_type')


# * Column is a 1-1 duplicate of payment column

# ### water_quality

# In[55]:


column_inspect('water_quality')


# In[56]:


abandoned = df.loc[(df['water_quality'] == 'fluoride abandoned') | (df['water_quality'] == 'salty abandoned')] 
abandoned.groupby('status_group').count()


# In[57]:


unknown_water = df.loc[df['water_quality'] == 'unknown']
unknown_water.groupby('basin').count()


# In[58]:


basin_groundwater = df.loc[(df['source_class'] == 'groundwater') & (df['basin'] == "Rufiji")]
basin_groundwater.groupby('quality_group').count() 


# * Water Quality - the quality of the water
# * No NaN values 
# * 50,000 of the values in the data set are considered soft water which is good - indicates the over all water in Tanzania, when available is good 
# * Two of the values are 'abandonded' 
# * 1876 unknown values 

# ### quality_group

# In[59]:


column_inspect('quality_group')


# In[60]:


df.loc[df['quality_group'] == 'unknown']


# * Quality Group - Water Quality
# * Same unknown rows as water_quality without the abandoned values 

# ### quantity

# In[61]:


column_inspect('quantity')


# In[62]:


df.loc[df['quantity'] == 'unknown']


# * Quantity - the amount of water the pump produces 
# * enough', 'insufficient', 'dry', 'seasonal', 'unknown 
# * There are 789 unknown values (more unknown than water quality but all unknown water qualities also have unknown quantities)

# ### quantity_group

# In[63]:


column_inspect('quantity_group')


# * Quantity Group - the amount of water the pump produces 
# * This is a duplicate to the quantity column

# ### source

# In[64]:


column_inspect('source')


# * Source - The Source of Water 
# * The column contains both other and unknowns variables 

# ### source_type

# In[65]:


column_inspect('source_type')


# In[66]:


compare_columns(df=df, x="source_type", color='payment')  


# In[67]:


compare_columns(df=df, x="source_type", color='basin')  


# * Source Type - The Source of Water 
# * 'spring' 'rainwater harvesting' 'dam' 'borehole' 'other' 'shallow well' 'river/lake' 
# * Contains only other 

# ### source_class

# In[68]:


column_inspect('source_class')


# * Source Class - The Source of Water 
# * 'groundwater' 'surface' 'unknown' 
# * Most General 

# ### waterpoint_type

# In[69]:


column_inspect('waterpoint_type')


# * Waterpoint type - the kind of water point 
# * 'communal standpipe', 'communal standpipe multiple', 'hand pump', 'other','improved spring', 'cattle trough', 'dam' 
# * Communal standpipe multiple is unique to this column

# ### waterpoint_type_group

# In[70]:


column_inspect('waterpoint_type_group')


# * Waterpoint Type Group - the kind of water point (more general) 
# * Identical to waterpoint type without "communal standpipe multiple"

# ### public_meeting

# In[72]:


column_inspect('public_meeting')


# * Public Meeting - A public meeting was held 
# * The vast majority of data regardless of functioning status had a public meeting but of those that didn't functioning / non functioning is almost 50/50

# ### scheme_management

# In[73]:


column_inspect('scheme_management')


# In[74]:


df['scheme_management'].isna()


# * Scheme management- Who opperates the waterpoint 
# * There are 3877 nan values 

# ### scheme_name

# In[75]:


column_inspect('scheme_name')


# * Scheme Name - Who opperates a waterpoint 
# * 28166 Nans & 2696 unique values 

# ### permit

# In[76]:


column_inspect('permit')


# In[77]:


non_permited = df.loc[df['permit'] == False] 
print(len(non_permited['installer'].unique()))
print(len(df['installer'].unique()))


# In[78]:


print(len(non_permited['construction_year'].unique()))
print(len(df['construction_year'].unique()))


# In[79]:


compare_columns(df=df, x='basin', color='status_group')


# * Permit - water point has a permit 
# * 3056 nan values  
# * Most pumps are permited 
# * Non permited pumps have a higher fail rate 
# * About 40% of installers installed a pump without permit 
# * Non permitted strutures built in every year on record

# ### extraction_type

# In[80]:


column_inspect('extraction_type')


# * Extraction type - What type of extration a water point uses (most specific) 
# * 18 values 'gravity' 'submersible' 'swn 80' 'nira/tanira' 'india mark ii' 'other' 'ksb' 'mono' 'windmill' 'afridev' 'other - rope pump' 'india mark iii' 'other - swn 81' 'other - play pump' 'cemo' 'climax' 'walimi' 'other - mkulima/shinyanga' 
# * Non nans

# ### extraction_type_group 

# In[81]:


column_inspect('extraction_type_group')


# * Extraction type - What type of extration a water point uses (generalized) 
# * 13 values - 'gravity' 'submersible' 'swn 80' 'nira/tanira' 'india mark ii' 'other' 'mono' 'wind-powered' 'afridev' 'rope pump' 'india mark iii' 'other handpump' 'other motorpump' 
# * No NaN 
# * Highest rate of functioning comes from gravity 

# ### extraction_type_class

# In[82]:


column_inspect('extraction_type_class')


# * Extraction type - What type of extration a water point uses (most gerneralized) 
# * 7 values - 'gravity' 'submersible' 'handpump' 'other' 'motorpump' 'wind-powered''rope pump'
# * No NaN 

# ### management

# In[83]:


column_inspect('management')


# *  Management - How the waterpoint is managed 
# * 12 values -'vwc' 'wug' 'other' 'private operator' 'water board' 'wua' 'company' 'water authority' 'parastatal' 'unknown' 'other - school' 'trust' 
# * There is a lot of overlap with different government agenciesn 
# * no nan values

# ### management_group

# In[84]:


column_inspect('management_group')


# *  Management_group - How the waterpoint is managed * most gerneral
# * 5 values - user-group' 'other' 'commercial' 'parastatal' 'unknown'
# * no nan values

# ### construction_year

# In[85]:


column_inspect('construction_year')


# In[86]:


df.loc[df['construction_year'] == 0]


# In[87]:


non_0 = df.loc[df['construction_year'] != 0]  
non_0


# In[88]:


fig = px.histogram(non_0, x='construction_year')  
fig.update_layout(barmode='group')
fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)', marker_line_width=1.5, opacity=0.6) 
fig.show() 


# * Construction year - The year the waterpoint was constructed 
# * No Nan values but 20709 years listed as 0 
# * Earliest True construction year 1964 
# * The majority of the water pumps were constructed in the mid 2000s 

# ## Investigate Each Continous Column 

# ### amount_tsh

# In[89]:


column_inspect('amount_tsh')


# In[90]:


df.groupby('amount_tsh')['id'].count().sort_values(ascending=False)


# In[91]:


df['amount_tsh'].describe()


# In[92]:


df.loc[df['amount_tsh'] > 0]


# * amount_tsh - Total static head  
# * 41639 listed as 0 - no way of knowing if this is the true tsh or if this is the unknown values 
# * Significganly less occurances for the non 0 values 
# * 17,761 non 0 values 

# ### gps_height

# In[93]:


column_inspect('gps_height')


# * gps height - Altitude of the well 
# * various points correlate to altitudes of wells ranging from -0 to over 2500

# ### longitude

# In[94]:


column_inspect('longitude')


# * logitude - GPS Coodinates 
# * Most points between 29 -40  
# * 1812 listed as 0.1 outliers 

# ### latitude

# In[95]:


column_inspect('latitude')


# * latitude - GPS Coodinates 
# * Points without longitude also missing latitude

# ### num_private

# In[96]:


column_inspect('num_private')


# * Non definition of this variable was found 
# * No pattern in the values

# ### population

# In[97]:


column_inspect('population')


# In[98]:


df.loc[df['population'] == 0]


# * Population - the population around the well 
# * 21381 listed as no population around the well 
