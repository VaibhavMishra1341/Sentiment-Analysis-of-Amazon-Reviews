# Amazon Reviews Analysis

This project focuses on analyzing Amazon product reviews using Python. It involves data visualization, missing value handling, sentiment analysis, and dynamic plotting using various libraries like Plotly, Matplotlib, TextBlob, and VADER Sentiment.

## Table of Contents

- [Prerequisites ⚙️](#prerequisites)
- [Installation 🛠️](#installation)
- [Usage 🚀](#usage)
- [Code Overview 📝](#code-overview)
- [Results 📊](#results)
- [Contact Me 👥](#contributing)
- [License 📄](#license)

## Prerequisites ⚙️

- Python 3.x
- Required libraries (install using `pip`):
  - pandas
  - numpy
  - cufflinks
  - matplotlib
  - plotly
  - textblob
  - vaderSentiment

## Installation 🛠️

1. Clone the repository:
```bash
git clone https://github.com/VaibhavMishra1341/Sentiment-Analysis-of-Amazon-Reviews.git
cd Sentiment-Analysis-of-Amazon-Reviews
```

2. Install the required libraries:
```bash
pip install -r requirements.txt
```

3. Place your Amazon reviews dataset (CSV format) in the project directory.

## Usage 🚀

1. Open a terminal and navigate to the project directory.

2. Run the analysis script:
```bash
python analysis.py
```

3. The script will perform various analyses, generate plots, and output sentiment analysis results.

## Code Overview 📝

- `analysis.py`: Main script containing the analysis and visualization code.
- `requirements.txt`: List of required libraries.

### Data Preprocessing 
```python
# Sorting data by wilson_lower_bound in descending order
df = df.sort_values("wilson_lower_bound", ascending=False)
# Dropping unnecessary column
df.drop('Unnamed: 0', inplace=True, axis=1)

# Handling missing values
def missing_values_analysis(df):
    # Identify columns with missing values
    na_columns_ = [col for col in df.columns if df[col].isnull().sum() > 0]
    # Calculate missing value counts and ratios
    n_miss = df[na_columns_].isnull().sum().sort_values(ascending=True)
    ratio_ = (df[na_columns_].isnull().sum() / df.shape[0] * 100).sort_values(ascending=True)
    missing_df = pd.concat([n_miss, np.round(ratio_, 2)], axis=1, keys=['Missing Values', 'Ratio'])
    missing_df = pd.DataFrame(missing_df)
    return missing_df

```

### Sentiment Analysis
```python
# Performing sentiment analysis using TextBlob and VADER
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Applying TextBlob sentiment analysis
df[['polarity', 'subjectivity']] = df['reviewText'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))

# Applying VADER sentiment analysis
for index, row in df['reviewText'].iteritems():
    score = SentimentIntensityAnalyzer().polarity_scores(row)
    neg = score['neg']
    neu = score['neu']
    pos = score['pos']

    if neg > pos:
        df.loc[index, 'sentiment'] = 'Negative'
    elif pos > neg:
        df.loc[index, 'sentiment'] = 'Positive'
    else:
        df.loc[index, 'sentiment'] = 'Neutral'

```

### Creating Dynamic Plots
```python
# Creating dynamic plots using Plotly and Matplotlib
def categorical_var_summary(df, column_name):
    # Creating a subplot with a countplot and a pie chart
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Countplot', 'Percentage'),
                        specs=[[{"type": "xy"}, {'type': 'domain'}]])
    
    # Adding a countplot
    fig.add_trace(go.Bar(y=df[column_name].value_counts().values.tolist(),
                         x=[str(i) for i in df[column_name].value_counts().index],
                         # ... other settings ...
                  row=1, col=1)
    
    # Adding a pie chart
    fig.add_trace(go.Pie(labels=df[column_name].value_counts().keys(),
                         values=df[column_name].value_counts().values,
                         # ... other settings ...
                  row=1, col=2)

    # Updating layout and displaying the plot
    fig.update_layout(title={'text': column_name, 'y': 0.5, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
                      template='plotly_white')
    iplot(fig)

```

## Results 📊

- The script provides insights into Amazon reviews, including data distribution, missing value analysis, and sentiment analysis results.

![newplot2](https://github.com/VaibhavMishra1341/Sentiment-Analysis-of-Amazon-Reviews/assets/39896268/822caf3b-b32e-4867-a908-06c865af5250)


- Plots generated using Plotly and Matplotlib visualize data patterns and sentiment distribution.

![newplot](https://github.com/VaibhavMishra1341/Sentiment-Analysis-of-Amazon-Reviews/assets/39896268/8057e584-2f19-4ddc-a378-e0faac2d6aaa)

## Contact Me 👥

- Website - [My Portfolio](https://vaibhav-mishra.vercel.app/)
- Github - [@VaibhavMishra1341](https://github.com/VaibhavMishra1341)
- LinkedIn - [@VaibhavMishra](https://www.linkedin.com/in/vaibhav-mishra-vm/)
- Instagram - [@_vaibhav__mishra_](https://www.instagram.com/_vaibhav__mishra_)

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

