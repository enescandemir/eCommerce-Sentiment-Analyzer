# E-Commerce Product Review Sentiment Analysis Application

This project is a web-based sentiment analysis application that
processes product reviews collected from e-commerce platforms
(Hepsiburada, Trendyol) and classifies them as positive, negative or
neutral. The system provides an end-to-end review analysis
infrastructure, including modules for data scraping, data preprocessing,
manual labeling, automatic labeling and user management.

------------------------------------------------------------------------

## Features

### User Operations

-   User registration and login
-   Admin and regular user roles
-   User management through an admin panel

### Data Scraping

-   Platform selection (Hepsiburada or Trendyol)
-   Collecting product reviews using a category link
-   Automated navigation with Selenium
-   Saving data to a SQL Server database
-   Downloading data in CSV format

### Data Preprocessing

Available operations: 
- Lowercasing
- Removing numbers
- Removing punctuation
- Removing emojis
- Cleaning repeated characters
- Removing stopwords

### Manual Labeling

-   Assigning the following labels to reviews:
    -   0: Negative
    -   1: Neutral
    -   2: Positive
-   Saving labeled data to SQL Server
-   Downloading labeled data in CSV format

### Automatic Labeling

-   Uploading an unlabeled CSV file
-   Selecting one of the following models (all models are trained with a BERT tokenizer):
    -   CNN
    -   RNN
    -   LSTM
    -   GRU
-   Generating an automatically labeled CSV file

------------------------------------------------------------------------

## Technologies Used

### Backend

-   ASP.NET
-   SQL Server
-   Entity Framework

### Data Scraping

-   Python
-   Selenium WebDriver

### Deep Learning

-   PyTorch
-   BERT tokenizer (bert-base-turkish-cased)

### Frontend

-   HTML, CSS
-   Animated dynamic menu

------------------------------------------------------------------------

## Project Workflow

1.  The user registers and logs in.
2.  The user selects a platform and category link to collect reviews.
3.  Reviews are downloaded or sent to the preprocessing module.
4.  Cleaned data is labeled manually or automatically.
5.  Labeled data is exported as a CSV file.
