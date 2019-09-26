import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from flask import Flask, render_template, request, send_from_directory, jsonify, send_file, redirect
import requests

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/hasil', methods=['GET','POST'])
def hasil():
    df = pd.read_json('digimon.json')
    dfimg = df[['no','digimon','image']]
    dfimg['digimon'] = dfimg['digimon'].apply(lambda x:x.capitalize())
    dfre = df[['no','digimon','stage','type','attribute']]
    dfre['digimon'] = dfre['digimon'].apply(lambda x:x.capitalize())
    dfre['x']=dfre['stage']+'budiman'+dfre['type']+'budiman'+dfre['attribute']
    model = CountVectorizer(
        tokenizer = lambda i: i.split('budiman'),
        analyzer = 'word',
    )
    matrix = model.fit_transform(dfre['x'])
    dfmx = matrix.toarray()
    score = cosine_similarity(dfmx)
    body = request.form
    diginame = body['digimon'].capitalize()
    if diginame not in list(dfre['digimon']):
        return redirect ('/notfound')
    imgre = dfimg['image'][dfimg['digimon']==diginame.capitalize()].values[0]
    stgdigi = dfre['stage'][dfre['digimon']==diginame.capitalize()].values[0]
    typedigi = dfre['type'][dfre['digimon']==diginame.capitalize()].values[0]
    attdigi = dfre['attribute'][dfre['digimon']==diginame.capitalize()].values[0]
    
    likeIndex = dfre['no'][dfre['digimon'] == diginame.capitalize()].values[0]
    allDigi = list(enumerate(score[likeIndex]))
    for i in range(len(allDigi)):
        if int(allDigi[i][0]) != int(likeIndex):
            similardigi = (sorted(allDigi,key = lambda x: x[1],reverse = True))
    listrecom = []
    for i in range(len(similardigi)):
        if int(similardigi[i][0]) != int(likeIndex):
            listrecom.append(similardigi[i])
    newlist = []
    imgsrc = []
    stgoth = []
    typeoth = []
    attoth = []
    for i in range(len(listrecom[:6])):
        newlist.append(dfre['digimon'][dfre['no']==listrecom[i][0]].values[0])
        imgsrc.append(dfimg['image'][dfimg['no']==listrecom[i][0]].values[0])
        stgoth.append(dfre['stage'][dfre['no']==listrecom[i][0]].values[0])
        typeoth.append(dfre['type'][dfre['no']==listrecom[i][0]].values[0])
        attoth.append(dfre['attribute'][dfre['no']==listrecom[i][0]].values[0])
    
    return render_template(
        'responsefound.html', 
        imgsrc=imgsrc,
        imgre=imgre,
        diginame=diginame,
        newlist=newlist,
        stgdigi=stgdigi,
        typedigi=typedigi,
        attdigi=attdigi,
        stgoth=stgoth,
        typeoth=typeoth,
        attoth=attoth
        )

@app.route('/notfound')
def notfound():
    return render_template('notfound.html')

if __name__ == '__main__':
    app.run(debug = True)


# df = pd.read_json('digimon.json')
# df = df[['no','digimon','stage','type','attribute']]
# df['digimon'] = df['digimon'].apply(lambda x:x.capitalize())

# df['x']=df['stage']+'budiman'+df['type']+'budiman'+df['attribute']
# model = CountVectorizer(
#     tokenizer = lambda i: i.split('budiman'),
#     analyzer = 'word',
# )

# matrix = model.fit_transform(df['x'])
# digimon = model.get_feature_names()
# countdigi = len(digimon)
# dfmx = matrix.toarray()

# score = cosine_similarity(matrix)

# like = 'agumon'
# likeIndex = df['no'][df['digimon'] == like.capitalize()].values[0]

# allDigi = list(enumerate(score[likeIndex]))

# for i in range(len(allDigi)):
#     if int(allDigi[i][0]) != int(likeIndex):
#         similardigi = (sorted(allDigi,key = lambda x: x[1],reverse = True))

# listrecom = []
# for i in range(len(similardigi)):
#     if int(similardigi[i][0]) != int(likeIndex):
#         listrecom.append(similardigi[i])

# newlist = []
# for i in range(len(listrecom[:6])):
#     newlist.append(df['digimon'][df['no']==listrecom[i][0]].values[0])

# print(newlist)