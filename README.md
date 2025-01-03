### Sentiment Analysis Using SVM

---

## **Project Overview**
This project performs sentiment analysis on the IMDb Movie Reviews Dataset to classify reviews as **positive** or **negative**. The model is built using **Support Vector Machines (SVM)**, and the hyperparameters are optimized using **GridSearchCV** for improved performance.

---

## **Dataset**
- **Source**: [IMDb Dataset](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Columns**:
  - `review`: Textual reviews from IMDb.
  - `sentiment`: Sentiment labels (positive/negative).

---

## **Features**
1. **Text Preprocessing**:
   - Lowercasing
   - Removing punctuation
   - Stopword removal
   - Lemmatization

2. **Feature Extraction**:
   - TF-IDF Vectorization with `ngram_range=(1, 2)` to capture unigrams and bigrams.

3. **Model**:
   - Support Vector Machine (SVM) with hyperparameter tuning.

4. **Hyperparameter Tuning**:
   - **GridSearchCV** is used to optimize parameters like `C`, `kernel`, `gamma`, and `degree`.

---

## **Requirements**
Install the required Python libraries using:

```bash
pip install -r requirements.txt
```

**Libraries**:
- pandas
- numpy
- scikit-learn
- nltk

---

## **Usage**

### Clone the Repository
```bash
git clone https://github.com/yourusername/sentiment-analysis-svm.git
cd sentiment-analysis-svm
```

### Download Dataset
1. Download the IMDb Dataset from [Kaggle](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).
2. Place the dataset file (`IMDB Dataset.csv`) in the project directory.

### Run the Script
Execute the main script:

```bash
python sentiment_analysis.py
```

---

## **Hyperparameter Tuning**
The script includes grid search for tuning SVM parameters:

- **Parameters Tuned**:
  - `C`: Regularization parameter
  - `kernel`: Type of kernel ('linear', 'rbf')
  - `gamma`: Kernel coefficient
  - `degree`: Polynomial degree (for `poly` kernel)

---

## **Results**
The final model achieves high accuracy in classifying sentiments and generates a classification report.

**Sample Output**:
```
Best parameters found: {'C': 1, 'kernel': 'linear', 'gamma': 'scale', 'degree': 3}
Best score found: 0.89
```

---

## **Directory Structure**
```
sentiment-analysis-svm/
│
├── IMDB Dataset.csv          # Dataset file
├── sentiment_analysis.py     # Main script
├── requirements.txt          # List of dependencies
├── README.md                 # Project documentation
```

---

## **Future Enhancements**
1. Use advanced techniques like Transformers (e.g., BERT) for better accuracy.
2. Add functionality to classify neutral reviews.
3. Deploy the model as a web app using Flask or Streamlit.

---

## **License**
This project is licensed under the MIT License. See the LICENSE file for details.
