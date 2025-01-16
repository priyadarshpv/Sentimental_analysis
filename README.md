# **Sentiment Analysis Using SVM and BERT**

---

## **Project Overview**
This project performs sentiment analysis on the **IMDb Movie Reviews Dataset** to classify reviews as **positive**, **negative**, or **neutral**. Two approaches are used for building the model:

1. **Support Vector Machines (SVM)**: Classical machine learning approach with hyperparameter tuning.
2. **BERT (Bidirectional Encoder Representations from Transformers)**: A state-of-the-art deep learning model for Natural Language Processing tasks.

---

## **Dataset**
- **Source**: [IMDb Dataset](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Columns**:
  - `review`: Textual reviews from IMDb.
  - `sentiment`: Sentiment labels (positive/negative).

---

## **Features**
### 1. **SVM Model**:
- **Text Preprocessing**:
  - Lowercasing
  - Removing punctuation
  - Stopword removal
  - Lemmatization
- **Feature Extraction**:
  - TF-IDF Vectorization with `ngram_range=(1, 2)` to capture unigrams and bigrams.
- **Model**:
  - Support Vector Machine (SVM) with optimized hyperparameters using **RandomizedSearchCV**.

### 2. **BERT Model**:
- Fine-tuned a pre-trained **BERT model** using the **Hugging Face Transformers library**.
- Text inputs are tokenized using **BERT Tokenizer** with a maximum sequence length of 128.
- The model predicts three classes: **positive**, **negative**, and **neutral**.
- Achieves high accuracy with the capability to handle complex language structures.

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
- transformers
- torch

---

## **Usage**

### Clone the Repository
```bash
git clone https://github.com/yourusername/sentiment-analysis-svm-bert.git
cd sentiment-analysis-svm-bert
```

### Download Dataset
1. Download the IMDb Dataset from [Kaggle](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).
2. Place the dataset file (`IMDB Dataset.csv`) in the project directory.

### Run the Scripts
1. **For SVM Model**:
   ```bash
   python sentiment_analysis_svm.py
   ```
2. **For BERT Model**:
   ```bash
   python sentiment_analysis_bert.py
   ```

---

## **BERT Model Implementation Details**
- **Pre-trained Model**: Fine-tuned `bert-base-uncased` for sentiment classification.
- **Loss Function**: Categorical Cross-Entropy.
- **Optimizer**: AdamW.
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score.
- **Training Framework**: TensorFlow/Keras with Hugging Face Transformers.

### Fine-tuning Process:
- Tokenization using **BERT tokenizer**.
- Inputs include:
  - `input_ids`
  - `attention_mask`
- Output probabilities are converted to class labels using `argmax`.

### Prediction:
- After training, the model can predict sentiments for new input text.

---

## **Results**
### SVM Model:
- **Hyperparameter Tuning**:
  - `C`: Regularization parameter
  - `kernel`: Type of kernel ('linear', 'rbf')
  - `gamma`: Kernel coefficient
  - `degree`: Polynomial degree (for `poly` kernel)
- **Best Parameters**:
  ```
  {'C': 1, 'kernel': 'linear', 'gamma': 'scale', 'degree': 3}
  ```
- **Accuracy**: ~89%

### BERT Model:
- **Fine-tuned Model**: `bert-base-uncased`.
- **Accuracy**: ~93%
- **Sample Output**:
  ```
  Input: "I absolutely love this product!"
  Predicted Label: positive
  ```

---

## **Directory Structure**
```
sentiment-analysis-svm-bert/
│
├── IMDB Dataset.csv            # Dataset file
├── sentiment_analysis_svm.py   # SVM model script
├── sentiment_analysis_bert.py  # BERT model script
├── requirements.txt            # List of dependencies
├── README.md                   # Project documentation
├── bert_sentiment_model/       # Saved fine-tuned BERT model
```

---

## **Future Enhancements**
1. Further fine-tuning of the BERT model with additional data.
2. Deployment as a web application using **Flask** or **Streamlit**.
3. Integration of additional advanced NLP techniques for feature extraction.
4. Add an ensemble method combining SVM and BERT predictions.

---

## **License**
This project is licensed under the MIT License. See the LICENSE file for details.

-----

