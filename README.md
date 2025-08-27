# 📖 Sentiment Analysis of Amazon Kindle Reviews  

This project applies **Natural Language Processing (NLP)** and **Machine Learning** to classify Amazon Kindle book reviews into **Positive**, **Neutral**, or **Negative** sentiments.  
We experimented with both **traditional ML models (SVM)** and **Transformer-based models (RoBERTa)** to evaluate performance.  

---

## 🚀 Project Overview
- **Goal**: Accurately classify Kindle book reviews based on text + metadata (star ratings).  
- **Dataset**: [Amazon Kindle Book Reviews for Sentiment Analysis](https://www.kaggle.com/datasets/meetnagadia/amazon-kindle-book-review-for-sentiment-analysis)
   
- **Models Used**:
  - Support Vector Machine (SVM)
  - RoBERTa (pre-trained & fine-tuned)

---

## 🛠️ Technologies
- **Python**  
- **Pandas / NumPy**  
- **Scikit-learn** (SVM)  
- **PyTorch** & **HuggingFace Transformers** (RoBERTa)  
- **Matplotlib / Seaborn** (evaluation & visualization)  

---

## 🔑 Approach
1. **Data Preprocessing**  
   - Lowercasing, punctuation & stopword removal  
   - Lemmatization and POS-tagging  
   - Balancing dataset to avoid bias  

2. **Model Training**  
   - SVM baseline with linear kernel  
   - RoBERTa (pre-trained weights)  
   - RoBERTa (fine-tuned on dataset)  

3. **Evaluation**  
   - Accuracy, Precision, Recall, F1-score  
   - Confusion Matrix for performance analysis  

---

## 📊 Results
| Model                   | Accuracy |
|--------------------------|----------|
| SVM (baseline)           | 63.0%    |
| RoBERTa (pre-trained)    | 55.6%    |
| RoBERTa (fine-tuned)     | **66.8%** |

✅ Fine-tuned RoBERTa outperformed baseline models, proving the value of transfer learning + domain-specific tuning.  

---

## 📥 How to Run
1. Open the notebook in **Google Colab**  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)  

2. Upload the dataset or connect to Kaggle via API  
3. Run the notebook cells step by step to train & evaluate models  

---

## 📑 Documentation
📄 Full analysis and results → [Project Report (PDF)](docs/Sentiment_Analysis_Reviewer.pdf)  

---

## 👤 Author
**Jeremy Lee Ming Heng**  
📧 leejeremy456@gmail.com  
🔗 [GitHub](https://github.com/JeremyL03)  
