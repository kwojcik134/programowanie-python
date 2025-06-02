import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def preprocess(df):
    # String columns -> {0, 1}
    data['Stage_fear'] = data['Stage_fear'].map({'Yes': 1, 'No': 0})
    data['Drained_after_socializing'] = data['Drained_after_socializing'].map({'Yes': 1, 'No': 0})
    df['Personality'] = df['Personality'].map({'Extrovert': 1, 'Introvert': 0})

    return df

def analyze(df):
    sns.set_theme(style = 'ticks', palette = 'cividis')

    #Distribution of target variable
    plt.figure()
    sns.countplot(df, x = df['Personality'], hue = df['Personality'], legend = False)
    plt.xlabel('')
    plt.ylabel('Count')
    plt.title('Distribution of Personality Types in Dataset')
    plt.show()

    #Correlation of variables

    plt.figure()
    sns.heatmap()
    plt.show()





#Reading file
data = pd.read_csv('C:/Users/RSM18/PycharmProjects/Pythom/data/personality_dataset.csv')

# Dropping missing values
data.dropna(inplace=True, ignore_index=True)

analyze(data)


