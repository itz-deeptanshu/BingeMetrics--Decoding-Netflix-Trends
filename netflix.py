import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')
os.makedirs('charts', exist_ok=True)
print("Charts folder ready!")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report)

sns.set_theme(style="whitegrid")  #Sets all seaborn charts to have a white background with grid lines 
plt.rcParams['figure.figsize'] = (12, 6) #Sets default figure size for all matplotlib charts to 12 inches wide and 6 inches tall
print("All libraries imported successfully!")

#==================================================
# LOADING THE DATA AND UNDERSTANDING IT
#==================================================
df = pd.read_csv('netflix_titles.csv')

print("Shape of dataset:", df.shape) #prints (rows,columns) thus tells how big the data is
print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
print("\nContent type count:")
print(df['type'].value_counts()) #prints the count of Movies vs TV shows in the dataset


print("=== Dataset Shape ===")
print(df.shape)

print("\n=== Data Types of Each Column ===")
print(df.dtypes)

print("\n=== Basic Statistics ===")
print(df.describe()) #Shows statistics for numeric columns: count, mean, min, max, standard deviation, etc.

print("\n=== Missing Values in Each Column ===")
print(df.isnull().sum()) #Counts missing values per column. isnull() returns True/False for each cell; .sum() adds up the Trues.

print("\n=== Missing Values in Percentage ===")
print(round(df.isnull().sum() / len(df) * 100, 2))

#===============================================
# DATA CLEANING AND PREPARATION
#===============================================

df['director'] = df['director'].fillna('Unknown')
df['cast'] = df['cast'].fillna('Unknown')
df['country'] = df['country'].fillna('Unknown')
df['rating'] = df['rating'].fillna('Unknown')
df['duration'] = df['duration'].fillna('Unknown')

print("Step 1 done - Missing values filled")

df.dropna(subset=['date_added'], inplace=True) #Removes entire rows where date_added is missing
print("Step 2 done - Dropped rows with missing date_added")

df['date_added'] = df['date_added'].str.strip() # Removes invisible leading/trailing spaces from dates to rpevent errors during conversion
print("Step 3 done - Stripped spaces from date_added")

df['date_added'] = pd.to_datetime(df['date_added'])
print("Step 4 done - Converted date_added to datetime")

df['year_added'] = df['date_added'].dt.year
df['month_added'] = df['date_added'].dt.month
print("Step 5 done - Extracted year_added and month_added")

before = len(df)
df.drop_duplicates(inplace=True) #Removes any identical duplicate rows. Counts before and after to tell you how many were removed.
after = len(df)
print(f"Step 6 done - Removed {before - after} duplicate rows")

print("\n=== Missing Values After Cleaning ===")
print(df.isnull().sum())

print("\n=== New Columns Preview ===")
print(df[['title', 'date_added', 'year_added', 'month_added']].head())

#==================================================
# EXPLORATORY DATA ANALYSIS (EDA): The Charts and Insights
#==================================================
type_counts = df['type'].value_counts()

#Q1 Movies vs TV shows
plt.figure(figsize=(8, 8)) #creates a blank chart canvas of 8x8
plt.pie(type_counts,labels=type_counts.index,autopct='%1.1f%%',colors=['#E50914', "#F1E9E9"],startangle=90,textprops={'fontsize': 14})
plt.title('Movies vs TV Shows on Netflix', fontsize=18, fontweight='bold', pad=20)
plt.tight_layout() #automatically adjusts the subplot parameters so that the subplots fit within the figure area
plt.savefig('charts/q1_movies_vs_tvshows.png', dpi=150)
plt.show() # displays the chart on the screen
print("Movies:", type_counts['Movie'])
print("TV Shows:", type_counts['TV Show'])

#Q2 Top 10 Countries
country_data = df['country'].str.split(',').explode().str.strip() #splits into a list and explode creates one row per country
country_data = country_data[country_data != 'Unknown'] #removes all rows with value 'Unknown'
top_countries = country_data.value_counts().head(10) #counts each country and takes the top 10
plt.figure(figsize=(12, 6))
bars = sns.barplot(x=top_countries.values,y=top_countries.index,palette='Reds_r')

for i, v in enumerate(top_countries.values):
    bars.text(v + 10, i, str(v), va='center', fontsize=11, fontweight='bold')

plt.title('Top 10 Countries by Content on Netflix', fontsize=18, fontweight='bold', color='#E50914')
plt.xlabel('Number of Titles', fontsize=13)
plt.ylabel('Country', fontsize=13)
plt.tight_layout()
plt.savefig('charts/q2_top_countries.png', dpi=150)
plt.show()

print("\n=== Top 10 Countries ===")
print(top_countries)

#Q3 Content added per year
yearly_content = df.groupby(['year_added', 'type']).size().reset_index(name='count')

#Groups data by year AND type, counts titles per group. Result: a table with columns year_added, type, count
plt.figure(figsize=(14, 6))
sns.lineplot(data=yearly_content,x='year_added',y='count',hue='type',marker='o',linewidth=2.5,palette={'Movie': '#E50914', 'TV Show': '#564d4d'})
#Draws two lines (one per type). hue='type' separates colors by type. marker='o' adds dots at each point.
for _, row in yearly_content.iterrows(): #Iterates through each row of the yearly_content table to add text labels with the count values above each point on the line chart.
    plt.text(row['year_added'],row['count'] + 15,str(row['count']),ha='center',fontsize=9,fontweight='bold')

plt.title('Content Added to Netflix Per Year', fontsize=18, fontweight='bold', color='#E50914')
plt.xlabel('Year', fontsize=13)
plt.ylabel('Number of Titles Added', fontsize=13)
plt.xticks(yearly_content['year_added'].unique(), rotation=45)
plt.legend(title='Content Type', fontsize=11)
plt.tight_layout()
plt.savefig('charts/q3_content_per_year.png', dpi=150)
plt.show()

print("\n=== Content Added Per Year ===")
print(yearly_content.pivot(index='year_added', columns='type', values='count'))



#Q4 Rating distribution
valid_ratings = ['G', 'PG', 'PG-13', 'R', 'NC-17', 'NR', 'UR','TV-Y', 'TV-Y7', 'TV-Y7-FV', 'TV-G', 'TV-PG', 'TV-14', 'TV-MA']

rating_counts = df[df['rating'].isin(valid_ratings)]['rating'].value_counts()

plt.figure(figsize=(12, 6))
bars = sns.barplot(x=rating_counts.index,y=rating_counts.values,palette='Reds_r')

for i, v in enumerate(rating_counts.values):
    bars.text(i, v + 10, str(v), ha='center', fontsize=11, fontweight='bold')

plt.title('Content Rating Distribution on Netflix', fontsize=18, fontweight='bold', color='#E50914')
plt.xlabel('Rating', fontsize=13)
plt.ylabel('Number of Titles', fontsize=13)
plt.xticks(fontsize=11)
plt.tight_layout()
plt.savefig('charts/q4_ratings.png', dpi=150)
plt.show()

print("\n=== Rating Counts ===")
print(rating_counts)

#Q5 Top genres
genre_data = df['listed_in'].str.split(',').explode().str.strip()

top_genres = genre_data.value_counts().head(15)

plt.figure(figsize=(12, 7))
bars = sns.barplot(x=top_genres.values,y=top_genres.index,palette='Reds_r')
for i, v in enumerate(top_genres.values):
    bars.text(v + 10, i, str(v), va='center', fontsize=11, fontweight='bold')

plt.title('Top 15 Most Popular Genres on Netflix', fontsize=18, fontweight='bold', color='#E50914')
plt.xlabel('Number of Titles', fontsize=13)
plt.ylabel('Genre', fontsize=13)
plt.tight_layout()
plt.savefig('charts/q5_genres.png', dpi=150)
plt.show()

print("\n=== Top 15 Genres ===")
print(top_genres)

#Q6 Movie duration distribution
movies_df = df[df['type'] == 'Movie'].copy()

movies_df['duration_int'] = movies_df['duration'].str.extract('(\d+)').astype(float)

plt.figure(figsize=(12, 6))
sns.histplot(data=movies_df,x='duration_int',bins=30,color='#E50914',edgecolor='white',linewidth=0.5)

avg_duration = movies_df['duration_int'].mean() #Calculates the average duration of movies by taking the mean of the duration_int column.
plt.axvline(avg_duration,color='#221F1F',linestyle='--',linewidth=2,label=f'Average: {avg_duration:.0f} mins')

plt.title('Movie Duration Distribution on Netflix', fontsize=18, fontweight='bold', color='#E50914')
plt.xlabel('Duration (minutes)', fontsize=13)
plt.ylabel('Number of Movies', fontsize=13)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('charts/q6_movie_duration.png', dpi=150)
plt.show()

print("\n=== Movie Duration Stats ===")
print(f"Average duration : {movies_df['duration_int'].mean():.0f} mins")
print(f"Shortest movie   : {movies_df['duration_int'].min():.0f} mins")
print(f"Longest movie    : {movies_df['duration_int'].max():.0f} mins")
print(f"Most common range: 80 - 120 mins")

#Q7 Content added per month
month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

monthly_counts = df['month_added'].value_counts().sort_index()
monthly_counts.index = monthly_counts.index.map(month_names)

plt.figure(figsize=(12, 6))
bars = sns.barplot( x=monthly_counts.index, y=monthly_counts.values, palette='Reds_r')

for i, v in enumerate(monthly_counts.values):
    bars.text(i, v + 10, str(v), ha='center', fontsize=11, fontweight='bold')

plt.title('Content Added to Netflix Per Month', fontsize=18, fontweight='bold', color='#E50914')
plt.xlabel('Month', fontsize=13)
plt.ylabel('Number of Titles Added', fontsize=13)
plt.tight_layout()
plt.savefig('charts/q7_content_per_month.png', dpi=150)
plt.show()

print("\n=== Content Added Per Month ===")
for month, count in zip(monthly_counts.index, monthly_counts.values):
    print(f"{month}: {count}")

#==================================================
# Feature Engineering: Preparing the data for machine learning
#==================================================
df['duration_int'] = df['duration'].str.extract('(\d+)').astype(float)
print("Step 1 done - Duration extracted as number")
print(df[['duration', 'duration_int']].head())

le_country = LabelEncoder()
le_rating = LabelEncoder()
df['country_encoded'] = le_country.fit_transform(df['country'])
df['rating_encoded'] = le_rating.fit_transform(df['rating'])
print("\nStep 2 done - Country and Rating encoded")
print(df[['country', 'country_encoded', 'rating', 'rating_encoded']].head())


le_type = LabelEncoder()
df['type_encoded'] = le_type.fit_transform(df['type'])
print("\nStep 3 done - Type encoded")
print(df[['type', 'type_encoded']].head())

X = df[['country_encoded', 'rating_encoded', 'duration_int', 'year_added', 'month_added', 'release_year']]
y = df['type_encoded']
print("\nStep 4 done - Features and target defined")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"\nFeature columns: {X.columns.tolist()}")
print(f"Target: type_encoded (0=Movie, 1=TV Show)")


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)
print("\nStep 5 done - Data split into train and test")
print(f"Training samples : {X_train.shape[0]}")
print(f"Testing samples  : {X_test.shape[0]}")

#==================================================
# MODEL TRAINING AND EVALUATION
#==================================================
model = RandomForestClassifier(n_estimators=100,max_depth=10,random_state=42)

print("Step 1 done - Model initialized")
print(f"Model: {model}")

print("\nStep 2 - Training the model...")
model.fit(X_train, y_train)
print("Step 2 done - Model trained successfully!")

y_pred = model.predict(X_test)
print("\nStep 3 done - Predictions made")
print(f"First 10 actual values    : {list(y_test[:10])}")
print(f"First 10 predicted values : {list(y_pred[:10])}")

print("\n=== Sample Predictions ===")
for actual, predicted in zip(list(y_test[:5]), list(y_pred[:5])): #pairs them together one by one(actual,predicted)
    actual_label = le_type.inverse_transform([actual])[0]
    predicted_label = le_type.inverse_transform([predicted])[0]
    status = '✓ Correct' if actual == predicted else '✗ Wrong'
    print(f"Actual: {actual_label:10} | Predicted: {predicted_label:10} | {status}")

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

print("=" * 45)
print(" MODEL EVALUATION RESULTS")
print("=" * 45)
print(f"  Accuracy  : {accuracy  * 100:.2f}%")
print(f"  Precision : {precision * 100:.2f}%")
print(f"  Recall    : {recall    * 100:.2f}%")
print(f"  F1 Score  : {f1        * 100:.2f}%")
print("=" * 45)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred,target_names=['Movie', 'TV Show']))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm,annot=True,fmt='d',cmap='Reds',xticklabels=['Movie', 'TV Show'],yticklabels=['Movie', 'TV Show'],linewidths=0.5,linecolor='white')
plt.title('Confusion Matrix', fontsize=18, fontweight='bold', color='#E50914')
plt.xlabel('Predicted Label', fontsize=13)
plt.ylabel('Actual Label', fontsize=13)
plt.tight_layout()
plt.savefig('charts/q8_confusion_matrix.png', dpi=150)
plt.show()

feature_names = X.columns.tolist()
importances = model.feature_importances_

plt.figure(figsize=(10, 6))
bars = sns.barplot( x=importances,y=feature_names,palette='Reds_r')

for i, v in enumerate(importances):
    bars.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=11, fontweight='bold')

plt.title('Feature Importance in Predicting Movie vs TV Show',fontsize=16, fontweight='bold', color='#E50914')
plt.xlabel('Importance Score', fontsize=13)
plt.ylabel('Feature', fontsize=13)
plt.tight_layout()
plt.savefig('charts/q9_feature_importance.png', dpi=150)
plt.show()

print("\n=== Feature Importance ===")
for name, score in sorted(zip(feature_names, importances),key=lambda x: x[1], reverse=True):
    print(f"{name:20} : {score:.4f}")

print("\n" + "=" * 45)
print("           PROJECT SUMMARY")
print("=" * 45)
print(f"  Total titles analysed : {len(df)}")
print(f"  Movies                : {len(df[df['type'] == 'Movie'])}")
print(f"  TV Shows              : {len(df[df['type'] == 'TV Show'])}")
print(f"  Features used         : {len(feature_names)}")
print(f"  Training samples      : {X_train.shape[0]}")
print(f"  Testing samples       : {X_test.shape[0]}")
print(f"  Model accuracy        : {accuracy * 100:.2f}%")
print("=" * 45)
print("\n✓ END OF PROJECT!")