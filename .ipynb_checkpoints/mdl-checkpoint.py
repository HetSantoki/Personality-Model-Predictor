import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


df = pd.read_csv("pd.csv")

stf = pd.get_dummies(df['Stage_fear'],drop_first=True,dtype=float)
das = pd.get_dummies(df['Drained_after_socializing'],drop_first=True,dtype=float)
target = pd.get_dummies(df['Personality'],drop_first=True,dtype=float)

df.drop(labels=["Stage_fear","Drained_after_socializing","Personality"],axis=1,inplace=True)
df["stf"]=stf
df["das"]=das
df["target"]=target



X = df.iloc[:,:-1]
y=df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
accuracy = accuracy_score(pred,y_test)
print(accuracy)
# with open("personality_model.pkl", "wb") as f:
#     pickle.dump(model, f)

