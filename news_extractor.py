from sklearn.utils import shuffle
import pandas as pd

df = pd.read_json(r'bbc/News_Category_Dataset_v2.json', lines=True)
df_crime = df[df["category"] == "CRIME"]
df_entertainment = df[df["category"] == "ENTERTAINMENT"]
df_politics = df[df["category"] == "POLITICS"]
df_commedy = df[df["category"] == "COMEDY"]
df_sport = df[df["category"] == "SPORTS"]
df_business = df[df["category"] == "BUSINESS"]
df_tech = df[df["category"] == "TECH"]
df_religion = df[df["category"] == "RELIGION"]

num_of_dataset = 2000
df_entertainment = df_entertainment[:num_of_dataset]
df_sport = df_sport[:num_of_dataset]
df_tech = df_tech[:num_of_dataset]
df_business = df_business[:num_of_dataset]
df_politics = df_politics[:num_of_dataset]


dataset3 = pd.concat([df_entertainment, df_sport, df_tech,
                      df_business, df_politics], ignore_index=True)

shuffleDataframe = shuffle(dataset3)

print(shuffleDataframe.head())
