from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

project_dir = Path(__file__).resolve().parent
data_path = project_dir / "data" / "creditcard.csv"
images_dir = project_dir / "images"
images_dir.mkdir(parents=True, exist_ok=True)

frame = pd.read_csv(data_path)

print(frame.shape)
print(frame.head())
frame.info()

print(frame["Class"].value_counts())
print(frame["Class"].value_counts(normalize=True)*100)

print(frame.isnull().sum())
print(frame["Amount"].describe())

train_df, remaining_df=train_test_split(frame,test_size=0.4,random_state=5,stratify=frame["Class"])
print(train_df.shape, remaining_df.shape)
print(train_df["Class"].value_counts(normalize=True)*100)
print(remaining_df["Class"].value_counts(normalize=True)*100)

validation_df, test_df=train_test_split(remaining_df,test_size=0.5,random_state=5,stratify=remaining_df["Class"])
print(validation_df.shape,test_df.shape)
print(train_df.groupby("Class")["Amount"].mean())

print(train_df.groupby("Class")["Amount"].describe())

class_counts=train_df["Class"].value_counts()
print(class_counts)

plt.bar(["Fraud","Normal"],[class_counts.loc[1],class_counts.loc[0]])
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")
plt.title("Training Set: Transaction Counts")
plt.savefig(images_dir / "class_distribution.png")
plt.show()

plt.figure()
plt.hist(train_df["Amount"],bins=50)
plt.title("Training Set: Transaction Amounts")
plt.xlabel("Transaction Amount")
plt.ylabel("Number of Transactions")
plt.savefig(images_dir / "amount_distribution.png")
plt.show()

condition = train_df["Amount"] <= 500
small_amount_df=train_df[condition]
print(small_amount_df.shape)

plt.figure()
plt.hist(small_amount_df["Amount"],bins=50)
plt.title("Training Set: Amounts up to 500")
plt.xlabel("Transaction Amount")
plt.ylabel("Number of Transactions")
plt.savefig(images_dir / "amount_distribution_under_500.png")
plt.show()

condition1=train_df["Class"]==0
normal_df=train_df[condition1]
fraud_df=train_df[~condition1]
print(normal_df.shape,fraud_df.shape)

amount_edges=list(range(0,501,20))

plt.figure()
plt.hist(normal_df["Amount"],bins=amount_edges,density=True,alpha=0.5,label="Normal")
plt.hist(fraud_df["Amount"],bins=amount_edges,density=True,alpha=0.5,label="Fraud")
plt.legend()
plt.title("Training Set: Amount Distribution by Class (0–500)")
plt.xlabel("Transaction Amount")
plt.ylabel("Probability Density")
plt.savefig(images_dir / "amount_by_class_under_500.png")
plt.show()

print(train_df["Time"].describe())
time_edges=list(range(0,172801,7200))

plt.figure()
plt.hist(normal_df["Time"],bins=time_edges,density=True,alpha=0.5,label="Normal")
plt.hist(fraud_df["Time"],bins=time_edges,density=True,alpha=0.5,label="Fraud")
plt.legend()
plt.title("Training Set: Time Distribution by Class")
plt.xlabel("Seconds Since First Transaction")
plt.ylabel("Probability Density")
plt.savefig(images_dir / "time_by_class.png")
plt.show()

print(train_df.groupby("Class")["V1"].describe())

v1_edges=list(range(-50,6,1))

plt.figure()
plt.hist(normal_df["V1"],bins=v1_edges,density=True,alpha=0.5,label="Normal")
plt.hist(fraud_df["V1"],bins=v1_edges,density=True,alpha=0.5,label="Fraud")
plt.title("Training Set: V1 Distribution by Class")
plt.xlabel("V1")
plt.ylabel("Probability Density")
plt.legend()
plt.savefig(images_dir / "v1_by_class.png")
plt.show()
