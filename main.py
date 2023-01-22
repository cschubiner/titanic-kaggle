from collections import defaultdict
from statistics import mode

import numpy as np
import torch.nn.functional as F
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

# Read in data from CSV file
data = pd.read_csv("titanic/train.csv")
# append in the test data
test_data = pd.read_csv("titanic/test.csv")
data = data.append(test_data, ignore_index=True)

# The csv is in the format of:
# PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
# 1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
# 3,1,3,"Heikkinen, Miss. Laina",female,26,0,0,STON/O2. 3101282,7.925,,S
# 4,1,1,"Futrelle, Mrs. Jacques Heath (Lily May Peel)",female,35,1,0,113803,53.1,C123,S

# One-hot encode categorical columns
# The columns we want to one-hot encode are ["Pclass", "Sex", "SibSp", "Parch", "Cabin", "Embarked"]
# We don't want to one-hot encode "PassengerId", "Name", "Ticket", "Fare", "Age"
# We don't want to one-hot encode "Survived" because that's our target column
# Cabin should be one-hot encoded, but it has too many unique values, so we'll include only the first letter, or "N" if it's null

# split Ticket into two columns: TicketCharacters (all a-Z anywhere in the string) and TicketNumbers (all 0-9)
data["TicketCharacters"] = data["Ticket"].apply(lambda x: "".join([c for c in x if c.isalpha()]))
data["TicketNumbers"] = data["Ticket"].apply(lambda x: "".join([c for c in x if c.isdigit()]))
data["TicketNumChars"] = data["TicketCharacters"].apply(lambda x: len(x))

# if TicketNumbers has any "" rows, fill them with 0
data["TicketNumbers"] = data["TicketNumbers"].replace("", 0)



for col_to_encode in ["Pclass", "Sex", "SibSp", "Parch", "Cabin", "Embarked", "TicketCharacters"]:
    if col_to_encode == "Cabin":
        data[col_to_encode] = data[col_to_encode].apply(lambda x: x[0] if pd.notnull(x) else "N")
    elif col_to_encode == "TicketCharacters":
        # take the first 2 characters only, sorted alphabetically
        data[col_to_encode] = data[col_to_encode].apply(lambda x: "".join(sorted(x)[:2]))
    one_hot = pd.get_dummies(data[col_to_encode], prefix=col_to_encode)
    # You ask, what is get_dummies? It's a function that takes a column of categorical data and returns a dataframe of one-hot encoded columns
    # For example, if you have a column of data ["A", "B", "C", "A", "B", "C", "A", "B", "C"]
    # get_dummies will return a dataframe with 3 columns, one for each unique value, and 1s and 0s in each row depending on whether the value in that row is the same as the column name
    # For example, if the column name is "A", then the column will be [1, 0, 0, 1, 0, 0, 1, 0, 0]
    # If the column name is "B", then the column will be [0, 1, 0, 0, 1, 0, 0, 1, 0]
    data = data.drop(col_to_encode, axis=1)
    data = data.join(one_hot)

# for the Age and Fare columns, fill out the null values with the mean of the column
data["Age"] = data["Age"].fillna(data["Age"].mean())
data["Fare"] = data["Fare"].fillna(data["Fare"].mean())

# convert the Name column to two columns, for the number of vowels and the number of consonants in the name
data["Name_vowels"] = data["Name"].apply(lambda x: len([c for c in x if c.lower() in "aeiou"]))
data["Name_consonants"] = data["Name"].apply(lambda x: len([c for c in x if c.lower() in "bcdfghjklmnpqrstvwxyz"]))
data = data.drop("Name", axis=1)
# drop the Ticket column
data = data.drop("Ticket", axis=1)

# for any other columns that are null, fill them with the mean of the column
for col in data.columns:
    if data[col].isnull().values.any():
        data[col] = data[col].fillna(data[col].mean())

# print out columns that have any cells with '' in them
for col in data.columns:
    if data[col].apply(lambda x: '' == str(x)).any():
        print(col)
        # fill them with the mean of the column
        data[col] = data[col].fillna(data[col].mean())

# convert all string columns to float columns
for col in data.columns:
    if data[col].dtype == "object":
        # if it has a . to it, then it's a float, else it's an int
        if "." in data[col].iloc[0]:
            data[col] = data[col].astype(float)
        else:
            data[col] = data[col].astype(int)

# normalize all columns to be between 0 and 1 using a min-max scaler
scaler = MinMaxScaler()
# don't transform the Survived column or the PassengerId column
data[data.columns.difference(["Survived", "PassengerId"])] = scaler.fit_transform(data[data.columns.difference(["Survived", "PassengerId"])])


# survived is the target column and can only be 0 or 1. Set all values that are not 0 or 1 to Null
data["Survived"] = data["Survived"].apply(lambda x: x if x in [0, 1] else None)

# Split data into train and test sets
# if there are any null values in the Survived column or it's not 0 or 1, then use those rows for the test set
# else, split the data into 80% train and 20% test
if data["Survived"].isnull().values.any():
    test = data[data["Survived"].isnull()]
    train = data[~data["Survived"].isnull()]
else:
    train, test = train_test_split(data, test_size=0.2)

# Split data into X and y
X_train = train.drop("Survived", axis=1)
y_train = train["Survived"]
X_test = test.drop("Survived", axis=1)
y_test = test["Survived"]

# Convert to tensors
# X_train = torch.tensor(X_train.values, dtype=torch.float32)
# To avoid a "TypeError: can't convert np.ndarray of type numpy.object_" error, we do the following:
X_train = torch.tensor(X_train.values.astype(np.float32))
y_train = torch.tensor(y_train.values.astype(np.float32))
X_test = torch.tensor(X_test.values.astype(np.float32))
y_test = torch.tensor(y_test.values.astype(np.float32))

# Create a 5 layer neural network using linear layers and relu activations between them
class Net(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc4 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc5 = torch.nn.Linear(hidden_size, output_size)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.sigmoid(self.fc5(x))
        return x


def test_accuracy_test():
    global output
    with torch.no_grad():
        output = net(X_test)
        output = output.round()
        correct = (output == y_test.view(-1, 1)).sum().item()
        print("Accuracy: {}".format(correct / len(y_test)))


passenger_id_to_predictions_dict = defaultdict(list)
# Train the network 5 times total, keeping track of the predictions
output = None
for train_iteration in range(7):
    net = Net(X_train.shape[1], 100, 1)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(40000):
        optimizer.zero_grad()
        output = net(X_train)
        loss = criterion(output, y_train.view(-1, 1))
        loss.backward()
        optimizer.step()
        if loss.item() < 0.13:
            break
        if epoch % 100 == 0:
            print("Epoch: {}, Loss: {}".format(epoch, loss.item()))
        if epoch % 2000 == 0:
            test_accuracy_test()

    # run the test set through the network to get the final Survived predictions
    with torch.no_grad():
        output = net(X_test)
        output = output.round()

    # now save the predictions to a CSV with only the PassengerId and Survived columns
    test["Survived"] = output.round().int().numpy()
    for index, row in test.iterrows():
        passenger_id_to_predictions_dict[row["PassengerId"]].append(row["Survived"])

# now that we have the predictions for each passenger, we can take the mode of the predictions
# and use that as the final prediction
passenger_id_to_final_prediction_dict = {}
count = 0
for passenger_id, predictions in passenger_id_to_predictions_dict.items():
    # take the average of the predictions, and round it to the nearest integer
    passenger_id_to_final_prediction_dict[passenger_id] = int(round(np.mean(predictions)))
    # if not all predictions are the same, print out the predictions and the id of the passenger, and the average
    if len(set(predictions)) > 1:
        print("Passenger ID: {}, Predictions: {}, Average: {}".format(passenger_id, predictions, np.mean(predictions)))
        count += 1

print("Number of passengers with different predictions: {}".format(count))
print(f"Num total passengers: {len(passenger_id_to_predictions_dict)}")

# now save the final predictions to a CSV with only the PassengerId and Survived columns
final_predictions = pd.DataFrame.from_dict(passenger_id_to_final_prediction_dict, orient="index")
final_predictions.reset_index(inplace=True)
final_predictions.columns = ["PassengerId", "Survived"]
# make the PassengerId column an integer
final_predictions["PassengerId"] = final_predictions["PassengerId"].astype(int)
final_predictions.to_csv("submission.csv", index=False)

# now call kaggle competitions submit -c titanic -f submission.csv -m "Message" on the command line to submit the predictions to Kaggle
import os
os.system("kaggle competitions submit -c titanic -f submission.csv -m \"Message\"")
