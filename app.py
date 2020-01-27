import pandas as pd
import nltk
import numpy as np
import sklearn as sk 
#import datetime
#import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression 


from sklearn import metrics

#from sklearn.pipeline import Pipeline
#from sklearn.model_selection import GridSearchCV


import os

from os import path
#from wordcloud import WordCloud

nb =  MultinomialNB()
count_vect = CountVectorizer()

#input data
df = pd.read_csv("call_data.csv")

#preprocessing

#text tonkenizer
from nltk.tokenize import sent_tokenize, RegexpTokenizer
tokenizer = RegexpTokenizer("[\w']+")

def word_tokenize(text):
    text = tokenizer.tokenize(text)
    return text

df["Call"] = df["Call"].apply(lambda x: word_tokenize(x))

#stopwords
from nltk.corpus import stopwords
stopwords = stopwords.words("english")

remove = ["inaudible", "overspeaking", "several", "severalinaudible", "noreply", "jessica", "secondsofsilence", "natasha", "grenfell", "tower", "towers", "okay", "yeah", "michelle", "severalinaudiblewords",
         "erm","mmm", "Greenfield", "hello", "debbie", "hamife", "susan", "sener", "mm", "hmm", "natasha", "anthony", "deb","antonio", "conversation", "surrey", "cordelia", "tony"]

for stopword in stopwords:
    remove.append(stopword)
    
    
#stemming
from nltk.stem import WordNetLemmatizer
#stemmer = SnowballStemmer("english", ignore_stopwords=True)

def stem_word(text):
    stemmed_words = []
    for word in text:
        if word not in remove:
            word = WordNetLemmatizer().lemmatize(word,'v')
            stemmed_words.append(word)
    return stemmed_words

df["Corpus"] = df["Call"].apply(lambda x: stem_word(x))
df["Call"] =df["Call"].apply(lambda x: ' '.join(x))
df["Corpus"] = df["Corpus"].apply(lambda x: ' '.join(x))


#Feature Extraction
y = df["Category"]
X = df["Corpus"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)


#TFIDF vectorizer

TF = TfidfVectorizer(
    min_df=4,
    smooth_idf = True,
    stop_words= remove,
    sublinear_tf=True,
    use_idf=True,
    strip_accents='unicode',
    analyzer=word_tokenize , 
    token_pattern=r'\w{2,}',  #vectorize 2-character words or more
    ngram_range=(1, 1),
    max_features=30000)

# fit and transform on it the training features
TF.fit(X_train)
X_train_word_features = TF.transform(X_train)

#transform the test features to sparse matrix
test_features = TF.transform(X_test)

# transform the holdout text for submission at the end
holdout_word_features = TF.transform(X)

df_tfidf = pd.DataFrame(holdout_word_features.toarray(), columns = TF.get_feature_names() )

train_target = y_train
test_target = y_test
classifier =  LogisticRegression(C=2,random_state = 1, class_weight = 'balanced')
classifier.fit(X_train_word_features, train_target)

feature_names=TF.get_feature_names()
    
#print(feature_names)

    
y_pred = classifier.predict(test_features)
y_pred_prob = classifier.predict_proba(test_features)
    


df["warning"] = classifier.predict(holdout_word_features)

    #print(confusion_matrix(test_target, y_pred))
#print(classification_report(test_target, y_pred))




import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
from dash_table import DataTable
import plotly.figure_factory as ff



#df['start_time'] = pd.to_datetime(df['start_time'])


dashboarddf = df[["Names","Call", "warning", "Category", "start_time", "GTIRT17", "count"]]

df = df[["Names","Call", "warning","start_time"]]
df_display = df[["Names", "Call", "warning"]]
dfcat = df.pivot_table(index = "start_time", columns = "warning",  aggfunc="size")
display_cat = dfcat.reset_index()


#dfcat["total"] = dfcat.sum(axis=1)

#df = df[["Names","Call", "warning","start_time"]]


#df["Flat_Number"] = ""
#df["Floor_Number"] = ""
#df["Fire"] = ""
#df["Smoke"] = ""
#df["People"] = ""

#list_of_calls = df['start_time'].unique().tolist()
#time_list = pd.to_datetime(time_list)
 #= ['All']+sorted(time_list)


#external_stylesheets = ['https://codepen.io/veradio/pen/KKwJEOv.css']
# external_stylesheets = external_stylesheets 

app = dash.Dash(__name__, )

colors = {
    'background': '##2C3C45',
    'text': '#7FDBFF'
}



app.layout = html.Div([
                html.Div([ 
                        html.Div([
                        dcc.Dropdown(id='cat_filter', multi=False,
                                     options=[{'label': i , 'value': i }
                                     for i in df['start_time'].unique()]),
    
                        ],style = {'width': '40%', 'display': 'inline-block'}),
                                                        
                    
                         DataTable(id='cat_table', columns=[{"name": i, "id": i} for i in display_cat.columns],
                                                   style_table={'maxHeight': '60px','overflow': 'scroll', 'width' : '100%'},
                                                   style_header={'backgroundColor': 'rgb(30, 30, 30)'}, 
                                                   style_cell={'backgroundColor': '#324A59','color': 'white', },
                                                   fixed_rows={ 'headers': True, 'data': 0 },
                                             
                                                   style_data = {'whiteSpace': 'normal','height': 'auto'},
                                                   data=dfcat.to_dict("rows")),
                                   ], style = {'width' : '100%','column-count': '2', 'column-width': 'initial', 'maxHeight': '60px'}),
                        
                        #),
                                            
                       
               
   
    
            html.Div([
                    html.Div(
                            dcc.Graph(id="my-graph", 
                                      figure = {'layout' : {'plot_bgcolor': 'rgb(50, 50, 50)', 'paper_bgcolor':'rgb(50, 50, 50)'}})
                            ),
                        
                            DataTable( id='adding-rows-table',
                                          columns=[{"name": "Fire" , "id" : "Fire"},
                                                  {"name": "Flat", "id": "Flat"},
                                                   {"name": "Floor", "id": "Floor"},
                                                   {"name": "People" , "id" : "People"},
                                                   {"name": "Smoke" , "id" : "Smoke"}],
                                
                                                    data=[
                                                            {'column-{}'.format(i): (j + (i-1)*5) for i in range(1, 5)}
                                                            for j in range(3)
                                                        ],
                                                        editable=True,
                                                        row_deletable=True, 
                                                        style_header_conditional=[
                                                        {'if': {'column_id': 'Fire'},
                                                            'backgroundColor': 'rgb(30, 30, 30)',
                                                            'color': '#F55B44'},
                                                        {'if': {'column_id': 'Flat'},
                                                            'backgroundColor': 'rgb(30, 30, 30)',
                                                            'color': '#56C28B'},
                                                        {'if': {'column_id': 'People'},
                                                            'backgroundColor': 'rgb(30, 30, 30)',
                                                            'color': '#DA5AF6'},
                                                        {'if': {'column_id': 'Smoke'},
                                                            'backgroundColor': 'rgb(30, 30, 30)',
                                                            'color': '#F6E762'},
                                                        {'if': {'column_id': 'Floor'},
                                                            'backgroundColor': 'rgb(30, 30, 30)',
                                                            'color': '#4B7FF4'},
                                                        ],
                                                        style_header={
                                                                'backgroundColor': 'rgb(30, 30, 30)',
                                                                'fontWeight': 'bold'},
                                                        sort_action="native",
                                                        filter_action="native",
                                                        style_cell = {'textAlign': 'left', 'fontWeight': 'bold', 'color' : '#243540' }
                                                    ),

                            html.Button('Add Row', id='editing-rows-button', n_clicks=0),]),

                        html.Div(
                                DataTable(id='table',
                                  columns=[{"name": i, "id": i} for i in df],
                                   #style_as_list_view=True,
                                   style_header={'backgroundColor': 'rgb(30, 30, 30)'}, 
                                   style_cell={'backgroundColor': '#324A59','color': 'white', 'maxWidth': '400px'},
                                   #style_data = {'whiteSpace': 'normal','height': 'auto'},
                                  #style_cell={'maxWidth': '400px', 'whiteSpace': 'normal'},
                                   editable=True,
                                  style_data_conditional=[  
                                          {'if': 
                                                              {'column_id': 'Call'},
                                                               #'backgroundColor': '#2E303A',
                                                               'color': 'white', 'whiteSpace': 'normal','height': 'auto'},
                    #                                         {'if': 
                    #                                          {'column_id': 'Call', 
                    #                                           'filter_query': '{Names} eq "CALLER:"'},
                    #                                           'backgroundColor': '#2E303A',
                    #                                           'color': 'white'},
                    #                                        {'if':
                    #                                           {'column_id': 'Names', 
                    #                                           'filter_query': '{Names} eq "OPERATOR:"'},
                    #                                           'backgroundColor': '#C4C3C7',
                    #                                           'color': 'white'},
#                                                            {'if':
#                                                               {'column_id': 'Call', 
#                                                               'filter_query': '{Names} eq "OPERATOR:"'},
#                                                                 'opacity': 0.6},
    
                                                 
                                                             {'if': 
                                                              {'column_id': 'warning',
                                                               'filter_query': '{warning} eq "People"'},
                                                               'color': '#DA5AF6'},
                                                            {'if': 
                                                              {'column_id': 'Call',
                                                               'filter_query': '{warning} eq "People"'},
                                                               'color': '#DA5AF6'},
                                  
                                                            {'if': 
                                                              {'column_id': 'warning', 
                                                               'filter_query': '{warning} eq "Fire"'},
                                                               'color': '#F55B44'},
                                                            {'if': 
                                                              {'column_id': 'Call', 
                                                               'filter_query': '{warning} eq "Fire"'},
                                                               'color': '#F55B44'},
                                                            {'if':
                                                               {'column_id': 'warning', 
                                                               'filter_query': '{warning} eq "Smoke"'},
                                                               'color': '#F6E762'},
                                                              {'if':
                                                               {'column_id': 'Call', 
                                                               'filter_query': '{warning} eq "Smoke"'},
                                                               'color': '#F6E762'},
                                                             {'if':
                                                               {'column_id': 'warning', 
                                                               'filter_query': '{warning} eq "Floor"'},
                                                               'color': '#4B7FF4'},
                                                            {'if':
                                                               {'column_id': 'Call', 
                                                               'filter_query': '{warning} eq "Floor"'},
                                                               'color': '#4B7FF4'},
                                                            {'if':
                                                               {'column_id': 'warning', 
                                                               'filter_query': '{warning} eq "Flat"'},
                                                               'color': '#56C28B'},
                                                             {'if':
                                                               {'column_id': 'Call', 
                                                               'filter_query': '{warning} eq "Flat"'},
                                                               'color': '#56C28B'}
                                                               
                                                               ], 
                                          data=df.to_dict("rows")),
                        
                style = {'color': 'rgb(30, 30, 30)' }),
]) 
    

@app.callback(
    Output('adding-rows-table', 'data'),
    [Input('editing-rows-button', 'n_clicks')],
    [State('adding-rows-table', 'data'),
     State('adding-rows-table', 'columns')])
def add_row(n_clicks, rows, columns):
    if n_clicks > 0:
        rows.append({c['id']: '' for c in columns})
    return rows


@app.callback([
    Output("my-graph", "figure"),
    Output('cat_table', 'data'),
    Output('table', 'data')],
    [Input('cat_filter', 'value')])


def update_table(inputx):
    if inputx:
        dff = df[df['start_time'] == inputx]
        #dff = dff[["Names", "Call", "warning"]]
        dfcatf = dfcat[dfcat.index == inputx ]
        dfcatf = dfcatf.fillna(0)
        display_cat = dfcatf #.reset_index()
       
        unstack = pd.melt(dfcatf,  
                          value_vars=list(dfcatf.columns), # list of days of the week
                          var_name='Column', 
                          value_name='Sums')
                          #
        #unstack = unstack.fillna(0)
        piedata = go.Pie(labels=unstack["Column"], sort=False, values=unstack["Sums"], marker = dict(colors = ['#F55B44', '#56C28B', '#4B7FF4', '#DA5AF6', '#F6E762', '#8F9DA6' ]))
#        layoutfig = layout('layoutplot_bgcolor': 'rgb(50, 50, 50)', 'paper_bgcolor':'rgb(50, 50, 50)')
        #trace = go.Pie(labels=unstack["Column"].unique().tolist(),  values=unstack["Sums"].tolist())
        
        return  {"data":[piedata], "layout" :{'layoutplot_bgcolor': '#324A59', 'paper_bgcolor':'#324A59', 'coloraxis' : 'showscale','font' :{ "color" : "white"}}}, display_cat.to_dict('rows'), dff.to_dict('rows')
                
                
                #go.Pie(unstack, values = unstack, names = "Column")  }, dfcatf.to_dict('rows'), dff.to_dict('rows')
    else:
        tot_cat = pd.DataFrame(dfcat.sum(),columns = ["sum"])
        tot_cat = tot_cat.reset_index()
        piefdata = go.Pie(labels= tot_cat["warning"],  values=tot_cat["sum"], sort=False, marker = dict(colors = ['#F55B44', '#56C28B', '#4B7FF4', '#DA5AF6', '#F6E762', '#8F9DA6' ]))
        dfcati = dfcat.reset_index()
        return {"data":[piefdata], "layout" :{'layoutplot_bgcolor': '#324A59', 'paper_bgcolor':'#324A59', 'font' :{ "color" : "white"} }},dfcati.to_dict('rows'), df.to_dict('rows')



if __name__ == '__main__':
    app.run_server(debug=True)