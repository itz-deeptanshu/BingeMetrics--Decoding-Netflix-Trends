# BingeMetrics: Decoding Netflix Trends

This project performs an end-to-end data science pipeline on a comprehensive Netflix catalog dataset from Kaggle. The objective is twofold: to uncover streaming content patterns through deep Exploratory Data Analysis (EDA), and to build a predictive classification model determining whether a given title should be classified as a TV Show or a Movie.

📁 Project Structure

BingeMetrics/

|── netflix_titles.csv          # Raw dataset

├── netflix_titles.ipynb           # Main analysis notebook

├── charts/                        # Exported plots

│   ├── q1_movies_vs_tvshows.png

│   ├── q2_top_countries.png

│   ├── q3_content_per_year.png

│   └── ...

└── README.md


📊 Dataset


Property                   Details

Source               Kaggle – Netflix Movies and TV Shows

Records              ~8,800 titles

Time Range           1925 – 2021

Features             show_id, type, title, director, cast, country, date_added, release_year, rating, duration, listed_in, description


🔍 Analysis Covered


1. Data Cleaning

── Handling null values across director, cast, country, date_added, and rating

── Parsing date_added into datetime format

── Extracting month_added and year_added as features


2. Content Distribution

── Movies vs. TV Shows split

── Year-wise content addition trends

── Monthly addition patterns (which months Netflix adds the most)


3. Geographic Analysis

── Top 10 countries producing Netflix content

── Country-wise breakdown of Movies vs. TV Shows

4. Genre / Category Analysis

── Most common genres (via listed_in column parsing)

── Genre overlap using frequency plots

5. Rating Analysis

── Distribution of content ratings (TV-MA, PG-13, R, etc.)

── Rating breakdown by content type

6. Duration Analysis

── Movie runtime distribution (minutes)

── TV Show season count distribution

7. Director & Cast Insights

── Top 10 most featured directors

── Top 10 most cast actors/actresses

📈 Key Findings

── ~70% of Netflix's catalog consists of Movies, with TV Shows making up the rest.
── The United States, India, and the United Kingdom are the top three content-producing countries.
── Netflix saw its largest content addition spike between 2017–2019.
── TV-MA is the most common rating, indicating a strong lean toward adult-oriented content.
── International Movies and Dramas are the most frequent genres on the platform.
── Most movies fall in the 80–120 minute runtime range.

🛠️ Tech Stack

Tool                                       Purpose
Python 3.10+                            Core language                           
Pandas                                  Data loading, cleaning, wrangling       
Matplotlib                              Base plotting                          
Seaborn                                 Statistical visualizations              
Jupyter                                 Interactive notebook environment

📄 License

This project is licensed under the MIT License.

🙋‍♂️ Author
Deeptanshu Chakraborty
AI & Data Science Engineering Student 
Thapar Institute of Engineering & Technology
