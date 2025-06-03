import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


NUMERIC_VARIABLES = ['Time spent Alone', 'Social event attendance', 'Going outside', 'Friends circle size',
                     'Post frequency']

def preprocess(df):
    # String columns -> {0, 1}
    df['Stage fear'] = df['Stage fear'].map({'Yes': 1, 'No': 0})
    df['Drained after socializing'] = df['Drained after socializing'].map({'Yes': 1, 'No': 0})
    df['Personality'] = df['Personality'].map({'Extrovert': 1, 'Introvert': 0})

    return df

def load_and_clean(path):
    #Load data into df
    data = pd.read_csv(path)
    df = data.copy()

    #Get rid of missing values
    df.dropna(inplace=True, ignore_index=True)

    #Change column names - looks better for boxplot titles
    mapper = {col: col.replace('_', ' ') for col in df.columns.values}
    df.rename(mapper, axis=1, inplace=True)

    print(df.head(5))
    #Normalize numerical data
    for col in NUMERIC_VARIABLES:
        min_val = min(df[col])
        max_val = max(df[col])
        df[col] = (df[col] - min_val)/(max_val - min_val)

    print(df.head(5))

    return df

def analyze(df):
    sns.set_theme(style = 'ticks', palette = 'cividis')

    #

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
        plt.title(f'{num} by Personality')
        plt.ylabel('')
    plt.tight_layout()
    plt.show()

    #Correlation of variables

    plt.figure(figsize = (10, 10))
    sns.heatmap(df.corr(numeric_only = True), annot = True, cmap = 'cividis')
    plt.show()

data = load_and_clean('C:/Users/RSM18/PycharmProjects/Pythom1/programowanie-python/data/personality_dataset.csv')

analyze(data)


