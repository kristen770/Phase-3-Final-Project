def plot_feature_importances(model):
    n_features = data_train.shape[1]
    plt.figure(figsize=(8,8))
    plt.barh(range(n_features), model.feature_importances_, align='center') 
    plt.yticks(np.arange(n_features), data_train.columns.values) 
    plt.xlabel('Feature importance')
    plt.ylabel('Feature') 

def base_model(target, predictors):  
    ("""Input: target, predictors 
     Runs through a basic 'regular' tree model for baseline 
     Output: feature importance plot, confustion matrix, classification report""")
    data = pd.get_dummies(predictors) 
    data_train, data_test, target_train, target_test = train_test_split(data, target,test_size = 0.25, random_state=123)
    tree_clf = DecisionTreeClassifier(criterion='gini', max_depth=5) 
    tree_clf.fit(data_train, target_train) 
    tree_clf.feature_importances_  
    pred = tree_clf.predict(data_test)
    print(confusion_matrix(target_test, pred))
    print(classification_report(target_test, pred))
    print("Testing Accuracy for Decision Tree Classifier: {:.4}%".format(accuracy_score(target_test, pred) * 100)) 

def column_inspect(df, colname):  
    import matplotlib.pyplot as plt 
    import seaborn as sns
    import plotly.express as px 
    import plotly.graph_objects as go 
    from plotly.subplots import make_subplots  
    
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

def set_dtypes(df):  
    ###This function takes in the Tanzania waterpump dataframe and set each column to their nessesary dtype###  
    
    df[['id', 'construction_year']] = df[['id', 'construction_year']].astype(int)  
    df[['amount_tsh', 'gps_height', 'longitude',
        'latitude', 'num_private', 'population']] = df[['amount_tsh', 'gps_height', 'longitude', 'latitude', 
                                                        'num_private', 'population']].astype(float) 
    df[['date_recorded', 'funder', 'installer','wpt_name', 'basin', 'subvillage', 'region', 'region_code', 
        'district_code', 'lga', 'ward', 'public_meeting', 'recorded_by', 'scheme_management', 'scheme_name', 
        'permit', 'extraction_type', 'extraction_type_group', 'extraction_type_class', 'management', 
        'management_group', 'payment', 'payment_type', 'water_quality', 'quality_group', 'quantity', 
        'quantity_group', 'source', 'source_type', 'source_class', 'waterpoint_type', 'waterpoint_type_group', 
        'status_group']] = df[['date_recorded', 'funder', 'installer', 'wpt_name', 'basin', 'subvillage', 'region',
                               'region_code','district_code', 'lga', 'ward', 'public_meeting', 'recorded_by',  
                               'scheme_management', 'scheme_name', 'permit', 'extraction_type', 
                               'extraction_type_group', 'extraction_type_class', 'management', 'management_group', 
                               'payment', 'payment_type', 'water_quality', 'quality_group', 'quantity', 
                               'quantity_group', 'source', 'source_type', 'source_class', 'waterpoint_type', 
                               'waterpoint_type_group', 'status_group' ]].astype(str)  
    return df    
    

def transform_dataframe(df): 
    ###This function takes in the Tanzania waterpump dataframe and returns the processes dataframe needed for modeling### 
    
    import numpy as np  
   
    #fill nan_values
    df['scheme_management'] == df['scheme_management'].fillna('unknown', inplace=True)  
    df['permit'] == df['permit'].fillna('unknown', inplace=True) 
    df['gps_height'] == df['gps_height'].replace(0, np.nan)
    df['funder'] == df['funder'].fillna('other', inplace=True) 
    df['public_meeting'] == df['public_meeting'].fillna('unknown', inplace=True)  
    df['population'] == df['population'].replace(0, np.nan) 
    df['longitude'] == df['longitude'].replace(0., np.nan) 
    df['latitude'] == df['latitude'].replace(-0.06, np.nan) 
    df['gps_height'] == df['gps_height'].replace(0, np.nan) 
    df['subvillage'].fillna('unknown', inplace=True)    
    
    #Takes the average of the watersource type to normalize average_tsh 
    df.groupby(['source_type', 'basin', 'waterpoint_type_group'])['amount_tsh'].transform('mean') 
    
    #take the average population of other pumps in the area 
    df['population'].fillna(df.groupby(['region', 'lga', 'ward', 'subvillage'])['population'].transform('mean'))
    
    #take the average longitude of other pumps in the area 
    df['longitude'].fillna(df.groupby(['region', 'lga', 'ward', 'subvillage'])['longitude'].transform('mean')) 

    #take the average latitude of other pumps in the area  
    df['latitude'].fillna(df.groupby(['region', 'lga', 'ward', 'subvillage'])['latitude'].transform('mean'))
    
    #classify know and unknown years
    df['construction_year'].mask(df['construction_year'] != 0, 'unknown', inplace=True) 
 
    #Groups scheme by type 
    df['scheme_management'].mask(df['scheme_management'] == 'Other', 'other', inplace=True)  
    df['scheme_management'].mask(df['scheme_management'] == 'Unkown', 'other', inplace=True) 
    df['scheme_management'].mask(df['scheme_management'] == 'VWC', 'private', inplace=True) 
    df['scheme_management'].mask(df['scheme_management'] == 'Private operator', 'private', inplace=True)  
    df['scheme_management'].mask(df['scheme_management'] == 'Private', 'private', inplace=True) 
    df['scheme_management'].mask(df['scheme_management'] == 'Biore', 'private', inplace=True)   
    df['scheme_management'].mask(df['scheme_management'] == 'Company', 'private', inplace=True)   
    df['scheme_management'].mask(df['scheme_management'] == 'Parastatal', 'private', inplace=True)   
    df['scheme_management'].mask(df['scheme_management'] == 'Trust', 'private', inplace=True)
    df['scheme_management'].mask(df['scheme_management'] == 'WUG', 'government', inplace=True)
    df['scheme_management'].mask(df['scheme_management'] == 'Water Board', 'government', inplace=True) 
    df['scheme_management'].mask(df['scheme_management'] == 'WUA', 'governement', inplace=True) 
    df['scheme_management'].mask(df['scheme_management'] == 'Water authority', 'government', inplace=True) 
    df['scheme_management'].mask(df['scheme_management'] == 'SWC', 'government', inplace=True)   
    df['scheme_management'].mask(df['scheme_management'] == 'None', 'other', inplace=True) 
    df['scheme_management'].mask(df['scheme_management'] == 'unknown', 'other', inplace=True) 
    
    #replace gps_height 0s with average  
    df['gps_height'].fillna(df.groupby(['region', 'lga', 'ward', 'subvillage'])['gps_height'].transform('mean'))
    
    
    #drop unnessesary columns
    df.drop(['wpt_name', 'installer', 'id', 'date_recorded', 'recorded_by',
            'payment_type', 'quality_group', 'quantity_group', 'source', 'source_class', 'waterpoint_type',
           'scheme_name', 'extraction_type_group', 'extraction_type_class', 'management_group', 'num_private'], axis=1, inplace=True)  
    
    return df 
        
def proccess_data(df):   
    ###This function takes in a dataframe and passes them through the processing functions it returns a dataframe###
    set_dtypes(df)
    transform_dataframe(df)  
    return df