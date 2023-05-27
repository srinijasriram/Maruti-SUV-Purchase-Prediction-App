import pickle

from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
clf = pickle.load(open("pmodel.pkl","rb"))
#1.reading the dataset
import pandas as pd
df = pd.read_csv("SUV_Purchase.csv")

#2.Feature Engineering
#df.drop(['A'],axis=1) i.e axis=1 for column and axis=0 for row
df = df.drop(['User ID',	'Gender'],axis=1)

#3.loading the data
X=df.iloc[:,:-1].values #iloc index location in 2d array
Y=df.iloc[:,-1:].values

#4.splitting the dataset
#train and test  data set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

app = Flask(__name__)

@app.route('/')

def hello():
    return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def predict_class():
    print([x for x in request.form.values()])
    features=[int(x) for x in request.form.values()]
    sst=StandardScaler().fit(x_train)
    out = clf.predict(sst.transform([features]))
    print(out)

    if out == 0:
        return render_template('index.html', pred=f'The person will not be able to buy the car')
    else:
        return render_template('index.html', pred=f'The person will be able to buy the car')

if __name__ == "__main__":
    app.run(debug=True)
