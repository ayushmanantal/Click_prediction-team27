#Importing all the libraries :
import pandas as pd
from missingpy import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer



#Loading the files:

df=pd.read_csv("C:/Users/ADMIN/PycharmProjects/untitled4/JabRefTrain.csv", low_memory=False)
df1=pd.read_csv("C:/Users/ADMIN/PycharmProjects/untitled4/JabRef(Test).csv",na_values = ['\\N','Withheld for privacy','nA','nan'],low_memory=False)

#Identified the columns with high Null values :
null_cnt = df.isnull().sum()
null_cnt = df.isnull().sum()
null_cnt = null_cnt[null_cnt!=0]
null_percent = null_cnt / len(df)
null_table = pd.concat([pd.DataFrame(null_cnt), pd.DataFrame(null_percent)], axis=1)
null_table.columns = ['counts', 'percentage']
null_table.sort_values('counts', ascending=False, inplace=True)
null_table = null_table[null_table['percentage']>=0.60]



#Imputation
df["query_document_id"].fillna("1111", inplace=True)
df1["query_document_id"].fillna("1111", inplace=True)


imputer = Imputer(missing_values="NaN", strategy="most_frequent", axis = 0)
df["recommendation_algorithm_id_used"]= imputer.fit_transform(df["recommendation_algorithm_id_used"].values.reshape(-1,1))
df["recommendation_algorithm_id_used"]=pd.DataFrame(df["recommendation_algorithm_id_used"])
df["year_published"]= imputer.fit_transform(df["year_published"].values.reshape(-1,1))
df["year_published"]=pd.DataFrame(df["year_published"])
df1["recommendation_algorithm_id_used"]= imputer.fit_transform(df1["recommendation_algorithm_id_used"].values.reshape(-1,1))
df1["recommendation_algorithm_id_used"]=pd.DataFrame(df1["recommendation_algorithm_id_used"])
df1["year_published"]= imputer.fit_transform(df1["year_published"].values.reshape(-1,1))
df1["year_published"]=pd.DataFrame(df1["year_published"])


df['local_time_of_request']=df['local_time_of_request'].fillna(method='ffill')
df['local_time_of_request']=df['local_time_of_request'].fillna(method='bfill')
df['local_hour_of_request']=df['local_hour_of_request'].fillna(method='ffill')
df['local_hour_of_request']=df['local_hour_of_request'].fillna(method='bfill')
df1['local_time_of_request']=df1['local_time_of_request'].fillna(method='ffill')
df1['local_time_of_request']=df1['local_time_of_request'].fillna(method='bfill')
df1['local_hour_of_request']=df1['local_hour_of_request'].fillna(method='ffill')
df1['local_hour_of_request']=df1['local_hour_of_request'].fillna(method='bfill')
df1["query_detected_language"].fillna("en", inplace=True)



#Separating the timesatmp column into Day,Month and Year :

df['local_time_of_request'] = pd.to_datetime(df.local_time_of_request)
df1['local_time_of_request'] = pd.to_datetime(df1.local_time_of_request)


df11 = df.local_time_of_request.dt.dayofweek
df11 = pd.DataFrame(df11)
df11.rename(columns={df11.columns[0]: "Day"}, inplace=True)
df12 = df.local_time_of_request.dt.month
df12 = pd.DataFrame(df12)
df12.rename(columns={df12.columns[0]: "Month"}, inplace=True)
df13 = df.local_time_of_request.dt.year
df13 = pd.DataFrame(df13)
df13.rename(columns={df13.columns[0]: "Year"}, inplace=True)
df = pd.concat([df, df11, df12, df13], axis=1)



df14 = df1.local_time_of_request.dt.dayofweek
df14 = pd.DataFrame(df14)
df14.rename(columns={df14.columns[0]: "Day"}, inplace=True)
df15 = df1.local_time_of_request.dt.month
df15 = pd.DataFrame(df15)
df15.rename(columns={df15.columns[0]: "Month"}, inplace=True)
df17 = df1.local_time_of_request.dt.year
df17 = pd.DataFrame(df17)
df17.rename(columns={df17.columns[0]: "Year"}, inplace=True)
df1 = pd.concat([df1, df14, df15, df17], axis=1)




# Reducing cardinality for contry_by_ip column
#def convert_sparse_values(df, col, threshold):
   # xxx = df[col].value_counts()[
  #      df[col].value_counts().cumsum() < df[col].value_counts().sum() * threshold].index.values
 #   df[col] = df[col].map(lambda x: x if x in xxx else 'other')


#convert_sparse_values(df, col='country_by_ip', threshold=.93)


#def convert_sparse_values(df5, col, threshold):
 #   xxx = df5[col].value_counts()[
#        df5[col].value_counts().cumsum() < df5[col].value_counts().sum() * threshold].index.values
#    df5[col] = df5[col].map(lambda x: x if x in xxx else 'other')


#convert_sparse_values(df5, col='country_by_ip', threshold=.85)



#Dropping the columns which are not contributing to the prediction :
df.drop(['recommendation_set_id',
         'user_id',
         'session_id',
         'query_identifier',
         'query_word_count',
         'query_char_count',
         'query_detected_language',
         'document_language_provided',
         'number_of_authors',
         'abstract_word_count',
         'abstract_char_count',
         'abstract_detected_language',
         'first_author_id',
         'num_pubs_by_first_author',
         'organization_id',
         'application_type',
         'item_type',
         'request_received',
         'response_delivered',
         'rec_processing_time',
         'app_version',
         'app_lang',
         'user_os',
         'user_os_version',
         'user_java_version',
         'user_timezone',
         'country_by_ip',
         'timezone_by_ip',
         'local_time_of_request', 'number_of_recs_in_set',
         'algorithm_class',
         'cbf_parser',
         'search_title',
         'search_keywords',
         'search_abstract',
         'time_recs_recieved',
         'time_recs_displayed',
         'time_recs_viewed',
         'clicks',
         'ctr','hour_request_received'
         ], axis=1, inplace=True)

df1.drop(['recommendation_set_id',
         'user_id',
         'session_id',
         'query_identifier',
         'query_word_count',
         'query_char_count',
         'query_detected_language',
         'document_language_provided',
         'number_of_authors',
         'abstract_word_count',
         'abstract_char_count',
         'abstract_detected_language',
         'first_author_id',
         'num_pubs_by_first_author',
         'organization_id',
         'application_type',
         'item_type',
         'request_received',
         'response_delivered',
         'rec_processing_time',
         'app_version',
         'app_lang',
         'user_os',
         'user_os_version',
         'user_java_version',
         'user_timezone',
         'country_by_ip',
         'timezone_by_ip',
         'local_time_of_request','number_of_recs_in_set',
         'algorithm_class',
         'cbf_parser',
         'search_title',
         'search_keywords',
         'search_abstract',
         'time_recs_recieved',
         'time_recs_displayed',
         'time_recs_viewed',
         'clicks',
         'ctr','hour_request_received'
         ], axis=1, inplace=True)


#Assigning Predictors and Target Value for training :

x = df.iloc[:, df.columns != 'set_clicked'].values
y = df.iloc[:, 4].values

x1 = df1.iloc[:, df1.columns != 'set_clicked'].values

#Train and Test Split:
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)


#Algorithm Applied to the Training data :
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf = clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

#Validation result :
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(len(df))
#Final prediction on actual test data:

y_final=clf.predict(x1)
y_final=pd.DataFrame(y_final)
y_final.to_csv("JabRef_Upload.csv")