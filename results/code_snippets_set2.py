import pandas as pd
from scipy.stats import iqr
from sklearn.metrics import mean_squared_error
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the feature set
feature_set = pd.read_csv('data_set2.csv')

# Completeness
print("Completeness:")
print(feature_set.isnull().sum())

# Consistency
print("\nConsistency:")
print("Correlation between 'total_claim_amount' and 'incident_severity':")
print(feature_set['total_claim_amount'].corr(feature_set['incident_severity']))
print("Correlation between 'age' and 'incident_severity':")
print(feature_set['age'].corr(feature_set['incident_severity']))

# Uniqueness
print("\nUniqueness:")
print("Number of duplicate rows:")
print(feature_set.duplicated().sum())

# Range Checks
print("\nRange Checks:")
print("Values outside the specified ranges:")
print(feature_set[~(feature_set['total_claim_amount'].between(100, 114920))])
print(feature_set[~(feature_set['incident_severity'].between(0, 3))])
print(feature_set[~(feature_set['age'].between(18, 64))])

# Outlier Detection
print("\nOutlier Detection:")
print("IQR for 'total_claim_amount':")
print(iqr(feature_set['total_claim_amount']))
print("IQR for 'incident_severity':")
print(iqr(feature_set['incident_severity']))
print("Potential outliers:")
print(feature_set[(feature_set['total_claim_amount'] < (feature_set['total_claim_amount'].quantile(0.25) - 1.5 * iqr(feature_set['total_claim_amount']))) |
                 (feature_set['total_claim_amount'] > (feature_set['total_claim_amount'].quantile(0.75) + 1.5 * iqr(feature_set['total_claim_amount'])))])
print(feature_set[(feature_set['incident_severity'] < (feature_set['incident_severity'].quantile(0.25) - 1.5 * iqr(feature_set['incident_severity']))) |
                 (feature_set['incident_severity'] > (feature_set['incident_severity'].quantile(0.75) + 1.5 * iqr(feature_set['incident_severity'])))])

# Category Validation for 'comments'
print("\nCategory Validation for 'comments':")
# Create controlled vocabulary
controlled_vocabulary = ['Accident', 'Injury', 'Medical Error', 'Other']
# Check for misspelled or invalid categories
print("Misspelled or invalid categories:")
print(feature_set['comments'][~feature_set['comments'].isin(controlled_vocabulary)])

# Text Quality Validation for 'comments'
print("\nText Quality Validation for 'comments':")
# Completeness: Check for comments with minimal or no content
print("Comments with less than 10 characters:")
print(feature_set['comments'][feature_set['comments'].str.len() < 10])
# Readability: Use Flesch-Kincaid Grade Level
# Install nltk and download the required package
nltk.download('punkt')
# Tokenize comments
feature_set['comments_tokenized'] = feature_set['comments'].apply(lambda x: word_tokenize(x.lower()))
# Remove stop words
feature_set['comments_tokenized'] = feature_set['comments_tokenized'].apply(lambda x: [word for word in x if word not in stopwords.words('english')])
# Calculate Flesch-Kincaid Grade Level
feature_set['flesch_kincaid_grade_level'] = feature_set['comments_tokenized'].apply(lambda x: nltk.corpus.gutenberg.sents(str(x)).flesch_kincaid_grade_level())
print("Comments with Flesch-Kincaid Grade Level above 12:")
print(feature_set[feature_set['flesch_kincaid_grade_level'] > 12]['comments'])
# Sentiment Analysis: Identify comments expressing positive, negative, or neutral sentiment
# Install TextBlob and download the required package
from textblob import TextBlob
# Calculate sentiment polarity
feature_set['sentiment_polarity'] = feature_set['comments'].apply(lambda x: TextBlob(x).sentiment.polarity)
print("Comments with negative sentiment:")
print(feature_set[feature_set['sentiment_polarity'] < 0]['comments'])
# Keyword Extraction: Extract relevant keywords from comments
vectorizer = CountVectorizer(stop_words='english')
X_train, X_test, y_train, y_test = train_test_split(feature_set['comments'], feature_set['incident_severity'], test_size=0.2, random_state=42)
vectorizer.fit(X_train)
X_train_tfidf = vectorizer.transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)
print("Keywords with the highest coefficients:")
print(sorted(zip(np.abs(classifier.coef_[0]), vectorizer.get_feature_names_out()), reverse=True)[:10])

# Feature Drift Detection
# Establish a baseline dataset with known characteristics
baseline_dataset = feature_set.iloc[:int(len(feature_set) * 0.8)]
# Calculate metrics for the baseline dataset
baseline_metrics = {
    'mean_total_claim_amount': baseline_dataset['total_claim_amount'].mean(),
    'mean_incident_severity': baseline_dataset['incident_severity'].mean(),
    'flesch_kincaid_grade_level': baseline_dataset['flesch_kincaid_grade_level'].mean(),
    'sentiment_polarity': baseline_dataset['sentiment_polarity'].mean()
}
# Periodically compare the current dataset with the baseline using statistical tests
current_dataset = feature_set.iloc[int(len(feature_set) * 0.8):]
# Calculate metrics for the current dataset
current_metrics = {
    'mean_total_claim_amount': current_dataset['total_claim_amount'].mean(),
    'mean_incident_severity': current_dataset['incident_severity'].mean(),
    'flesch_kincaid_grade_level': current_dataset['flesch_kincaid_grade_level'].mean(),
    'sentiment_polarity': current_dataset['sentiment_polarity'].mean()
}
# Perform t-tests and chi-square tests to detect significant differences
t_test_results = {}
chi_square_test_results = {}
for metric in baseline_metrics.keys():
    t_test_results[metric] = mean_squared_error(baseline_dataset[metric], current_dataset[metric])
    chi_square_test_results[metric] = 0  # Placeholder for chi-square test results
# Print the validation outcomes
print("\nValidation Outcomes:")
print(t_test_results)
print(chi_square_test_results)

# Save the validation outcomes to a CSV file
validation_results = pd.DataFrame({
    'Completeness_Null_Count': feature_set.isnull().sum(),
    'Consistency_Correlation_total_claim_amount_incident_severity': feature_set['total_claim_amount'].corr(feature_set['incident_severity']),
    'Consistency_Correlation_age_incident_severity': feature_set['age'].corr(feature_set['incident_severity']),
    'Uniqueness_Duplicate_Count': feature_set.duplicated().sum(),
    'Range_Checks_total_claim_amount_Outliers': feature_set[~(feature_set['total_claim_amount'].between(100, 114920))],
    'Range_Checks_incident_severity_Outliers': feature_set[~(feature_set['incident_severity'].between(0, 3))],
    'Range_Checks_age_Outliers': feature_set[~(feature_set['age'].between(18, 64))],
    'Outlier_Detection_total_claim_amount_IQR': iqr(feature_set['total_claim_amount']),
    'Outlier_Detection_total_claim_amount_Potential_Outliers': feature_set[(feature_set['total_claim_amount'] < (feature_set['total_claim_amount