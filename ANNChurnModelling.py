# Artificial Neural Network


# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('D:\\Learn & Projects\\DeepLearning\\Dataset\\Churn_ANN.csv')
#X = dataset.iloc[:, 3:13]
#y = dataset.iloc[:, 13]

#Create dummy variables
#geography=pd.get_dummies(X["Geography"],drop_first=True)
#gender=pd.get_dummies(X['Gender'],drop_first=True)
#dataset['Geography'] = dataset['Geography'].map({'France' : 0, 'Germany' : 1, 'Spain' : 2})
#dataset['Gender'] = dataset['Gender'].map({'Male' : 0, 'Female' : 1})

df_eda = dataset.iloc[:,3:14]

# Create a categorical column list where it can be converted to dummies
categorical_cols = list(dataset[['Geography','Gender', 'Tenure', 
                                 'NumOfProducts', 'HasCrCard','IsActiveMember']])
encodedDf_churn = pd.get_dummies(df_eda, columns = categorical_cols)
encodedDf_churn.head()

## Concatenate the Data Frames

X=encodedDf_churn.drop(columns='Exited')
y = encodedDf_churn['Exited']


## Drop Unnecessary columns
# X=X.drop(['Geography','Gender'],axis=1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

var1 = X_train
var2 = X_test


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from  tensorflow.keras.layers import LeakyReLU,PReLU,ELU
from  tensorflow.keras.layers import Dropout


# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer= 'he_uniform',activation='relu',input_dim = 28))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'he_uniform',activation='relu'))
# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'Adamax', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
model_history=classifier.fit(X_train, y_train,validation_split=0.33, batch_size = 10, epochs = 100)

# list all data in history

print(model_history.history.keys())
# summarize history for accuracy

plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Calculate the Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)
print(score*100,'%')


from tensorflow.keras.models import save_model
model_history.save('D:\\Learn & Projects\\DeepLearning\\Coding\\annChurnModel',save_format='pkl')















