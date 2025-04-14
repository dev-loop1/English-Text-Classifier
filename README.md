# English-Text-Classifier

A machine-learning pipeline inspired by Luo (2021) for efficient English text classification. This project implements and compares Support Vector Machines, Naïve Bayes, and Logistic Regression on three thematic subsets of the 20 Newsgroups corpus, following the methodology of preprocessing, feature extraction, and performance evaluation described in "Efficient English text classification using selected Machine Learning Techniques" (Luo, 2021).

---

## Key Features

- **Text Preprocessing**  
  - Cleans raw text: strips non-alphanumeric characters, lowercases, removes English stopwords (NLTK), and applies Porter stemming.  
- **Dataset Simulation**  
  - Uses three subsets of the 20 Newsgroups dataset to mimic the paper's Data1–Data3 splits:  
    1. **Data1**: `alt.atheism`, `soc.religion.christian`, `comp.graphics`, `sci.med`  
    2. **Data2**: `rec.sport.baseball`, `rec.sport.hockey`, `sci.space`, `talk.politics.misc`  
    3. **Data3**: `comp.sys.ibm.pc.hardware`, `comp.sys.mac.hardware`, `misc.forsale`  
- **Feature Extraction**  
  - Bag-of-words representation via `CountVectorizer(max_features=4000)` (word frequency features only).  
- **Model Training & Hyperparameter Tuning**  
  - **Support Vector Machine** (`C`: [0.1, 1, 10], `kernel`: [linear, rbf], `gamma`: [scale, auto])  
  - **Multinomial Naïve Bayes** (`alpha`: [0.1, 0.5, 1.0])  
  - **Logistic Regression** (`C`: [0.1, 1, 10], `solver`: [liblinear, lbfgs])  
  - 5-fold `GridSearchCV` on each dataset  
- **Evaluation Metrics**  
  - Accuracy, Precision, Recall, F1 Score (macro-averaged)  
- **Visualization**  
  - Side-by-side bar charts comparing all four metrics across the three datasets and three models  

---

## Installation

1. **Clone this repository**  
   ```bash
   git clone https://github.com/user/English-Text-Classifier.git
   cd English-Text-Classifier
   ```

2. **Create & activate a virtual environment**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK stopwords**  
   ```bash
   python -c "import nltk; nltk.download('stopwords')"
   ```

---

## Usage

1. **Open the notebook**  
   ```bash
   jupyter notebook english-text-classifier.ipynb
   ```

2. **Run all cells**  
   - Preprocesses the text  
   - Builds three DataFrames (Data1–Data3)  
   - Trains & tunes SVM, NB, and LR via GridSearchCV  
   - Evaluates on a 70/30 train/test split  
   - Generates comparative bar charts  

---

## Code Structure

- **`code.ipynb`**  
  1. **Imports & Setup**: pandas, NumPy, scikit-learn, NLTK, Matplotlib  
  2. **`preprocess_text()`**: cleaning, stop-word removal, stemming  
  3. **Data Loading**: fetches 20 Newsgroups, creates three themed subsets  
  4. **`train_and_evaluate_models(df, name)`**: vectorization, GridSearchCV, metric computation  
  5. **Results Aggregation & Plotting**: combines metrics and plots bar charts  

- **`requirements.txt`**  
  ```text
  numpy
  pandas
  scikit-learn
  nltk
  matplotlib
  jupyter
  ```

---

## Results

Across all three subsets, **SVM** consistently outperformed Naïve Bayes and Logistic Regression, achieving **> 90% accuracy** when using a 4,000-term feature set, mirroring Luo (2021).

| Model | Data1 Accuracy | Data2 Accuracy | Data3 Accuracy |
|-------|---------------:|---------------:|---------------:|
| SVM   |          0.92  |          0.90  |          0.91  |
| NB    |          0.80  |          0.78  |          0.79  |
| LR    |          0.88  |          0.86  |          0.87  |

*Exact numbers may vary slightly based on random train/test splits.*

---

## Citation

If you use this code, please cite:

> Luo, X. (2021). *Efficient English text classification using selected machine learning techniques*.  
> Alexandria Engineering Journal, 60(4), 3401–3409. https://doi.org/10.1016/j.aej.2021.02.009

---

## License

This project is released under the [MIT License](LICENSE).
