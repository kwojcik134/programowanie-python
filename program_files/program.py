import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


NUMERIC_VARIABLES = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size',
                     'Post_frequency']

STRING_VARIABLES = ['Stage_fear', 'Drained_after_socializing']

def preprocess(df):
    #Encoding yes/no data
    # enc = LabelEncoder()
    # to_encode = [col for col in STRING_VARIABLES]
    # to_encode.append('Personality')
    # for col in to_encode:
    #     enc.fit_transform(df[col].tolist())

    print(df.head(5))

    return df


def analyze(df):
    sns.set_theme(style = 'ticks', palette = 'cividis')

    #General data shape


    #Distribution of target variable
    plt.figure()
    sns.countplot(df, x = df['Personality'], hue = df['Personality'], legend = False)
    plt.xlabel('')
    plt.ylabel('Count')
    plt.title('Distribution of Personality Types in Dataset')
    plt.show()


    #Variables by personality
    plt.figure(figsize=(10, 10))
    for i, num in enumerate(NUMERIC_VARIABLES, 1):
        plt.subplot(3, 2, i)
        sns.boxplot(x = 'Personality', y = num, data = df, hue = 'Personality', legend = False)
        plt.title(f'{num.replace("_", " ")} by Personality')
        plt.ylabel('')
    plt.tight_layout()
    plt.show()

    #Correlation of variables

    plt.figure(figsize = (10, 10))
    sns.heatmap(df.corr(numeric_only = True), annot = True, cmap = 'cividis')
    plt.show()

path = 'C:/Users/RSM18/PycharmProjects/Pythom1/programowanie-python/data/personality_dataset.csv'
# Load data into df
raw_data = pd.read_csv(path)
data = raw_data.copy()

# Get rid of missing values
data.dropna(inplace=True, ignore_index=True)

preprocess(data)


