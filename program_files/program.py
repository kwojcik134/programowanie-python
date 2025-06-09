import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

path = 'C:/Users/RSM18/PycharmProjects/Pythom1/programowanie-python/data/Healthcare-Diabetes.csv'
# Load data into df
raw_data = pd.read_csv(path)
data = raw_data.copy()
data.drop('Id', axis = 1, inplace = True)
print(data.head())

#Basic data analysis
print(data.info())
print('Missing values:')
print(data.isnull().sum())

x_set = data.drop('Outcome', axis = 1)
y_set = data['Outcome']

# sns.set_theme(style = 'ticks', palette = 'deep')
#
# #Class distribution
# plt.figure()
# sns.countplot(x = y_set, data = data, hue = y_set, legend = False)
# plt.title('Class distribution of outcomes')
# plt.ylabel('Count')
# plt.xlabel('Outcome')
# plt.show()
#
# #Feature boxplots
# plt.figure(figsize = (10, 20))
# for i, col in enumerate(x_set.columns.tolist(), 1):
#     plt.subplot(3, 3, i)
#     sns.boxplot(data, x = y_set, y = col)
#     plt.title(f'{col} by outcome')
# plt.tight_layout()
# plt.ylabel('')
# plt.xlabel('')
# plt.show()

#Preprocessing
x_train, x_test, y_train, y_test = train_test_split(x_set, y_set, test_size = 0.2, random_state = 42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

#Linear SVC model
lin_svc = LinearSVC()

lin_svc.fit(x_train_scaled, y_train)
y_lsvc_predicted = lin_svc.predict(x_test_scaled)

lsvc_accuracy = accuracy_score(y_test, y_lsvc_predicted)
print(lsvc_accuracy)
lsvc_confusion = confusion_matrix(y_test, y_lsvc_predicted)
print(lsvc_confusion)
lsvc_rep = classification_report(y_test, y_lsvc_predicted)
print(lsvc_rep)




