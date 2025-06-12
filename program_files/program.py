import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

sns.set_theme(style = 'ticks', palette = 'deep')
path = 'C:/Users/RSM18/PycharmProjects/Pythom1/programowanie-python/data/Healthcare-Diabetes.csv'
if not os.path.exists(path):
    raise FileNotFoundError(f'The path: {path} does not exist. Please provide valid path')
if not path[-4:] == '.csv':
    raise Exception('Please provide a valid .csv files')
out_path = 'C:/Users/RSM18/PycharmProjects/Pythom1/programowanie-python/output/'
if not os.path.exists(out_path):
    raise FileNotFoundError(f'The path: {out_path} does not exist. Please provide valid path')

def evaluate(test, prediction, model_name):
    accuracy = accuracy_score(test, prediction)
    print(accuracy)
    confusion = confusion_matrix(test, prediction)
    print(confusion)
    report = classification_report(test, prediction)
    print(report)

    evaluation = (f'Accuracy: {accuracy}\n'
                  f'Classification report: {report}')

    with open(f'{model_name}_evaluation.txt', 'w') as file:
        file.write(evaluation)

    # Confusion matrix plot
    plt.figure()
    sns.heatmap(confusion, annot = True, cmap = 'crest', fmt = 'd', cbar=False,
            xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'{model_name}_matrix.jpg')

# Load data into df


raw_data = pd.read_csv(path)
data = raw_data.copy()
data.drop('Id', axis = 1, inplace = True)

#Basic data analysis
print(data.describe())
print(data.info())
print('Missing values:')
print(data.isnull().sum())

info = (f'{data.describe()}\n'
        f'{data.info()}\n'
        f'Missing values:\n'
        f'{data.isnull().sum()}')

os.chdir(out_path)

with open('info.txt', 'w') as f:
    f.write(info)

x_set = data.drop('Outcome', axis = 1)
y_set = data['Outcome']

#Class distribution
plt.figure()
sns.countplot(x = y_set, data = data, hue = y_set, legend = False)
plt.title('Class distribution of outcomes')
plt.ylabel('Count')
plt.xlabel('Outcome')
plt.savefig('distribution.jpg')

#Feature boxplots
plt.figure(figsize = (30, 30))
for i, col in enumerate(x_set.columns.tolist(), 1):
    plt.subplot(3, 3, i)
    sns.boxplot(data, x = y_set, y = col)
    plt.title(f'{col} by outcome')
plt.tight_layout()
plt.ylabel('')
plt.xlabel('')
plt.savefig('boxplots.jpg')

#Handling outliers
for col in x_set.columns:
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    q_diff = q3 - q1
    lower_bound = q1 - 1.5 * q_diff
    upper_bound = q3 + 1.5 * q_diff
    data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)

#Preprocessing
x_train, x_test, y_train, y_test = train_test_split(x_set, y_set, test_size = 0.2, random_state = 42, stratify = y_set)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

#Linear SVC model
lin_svc = LinearSVC()

lin_svc.fit(x_train_scaled, y_train)
y_lsvc_predicted = lin_svc.predict(x_test_scaled)

evaluate(y_test, y_lsvc_predicted, 'lin_svc')

#K Nearest Neighbors model

kneigh = KNeighborsClassifier()

kneigh.fit(x_train_scaled, y_train)
y_kneigh_predicted = kneigh.predict(x_test_scaled)

evaluate(y_test, y_kneigh_predicted, 'knn')

#Gradient Boosting model

gbc = GradientBoostingClassifier(random_state = 42, max_depth = 5)

gbc.fit(x_train_scaled, y_train)
y_gbc_predicted = gbc.predict(x_test_scaled)

evaluate(y_test, y_gbc_predicted, 'gbc')