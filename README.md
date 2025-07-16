# Malicious URL Detection using Machine Learning and NLP
Built by **Sujit Sanjay Deshmukh**

This project uses machine learning and natural language processing (NLP) to detect whether a given URL is safe or malicious. It’s powered by a Random Forest classifier and TF-IDF vectorization, trained on real-world phishing and benign URLs.

---

## Technologies Used
- Python 3.9+
- Pandas, NumPy
- Scikit-learn (Random Forest)
- TF-IDF (text feature extraction)
- Streamlit (web app)
- Pickle (model saving)

---

## Project Structure

```
├── app.py                 # Streamlit UI
├── train_model.py         # Model training script
├── rf_model.pkl           # Trained ML model
├── tfidf_vectorizer.pkl   # Saved TF-IDF vectorizer
├── malicious_phish.csv    # Dataset (from Kaggle)
└── README.md              # This file
```

---

## Dataset

Source: [Malicious URLs Dataset - Kaggle](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset)

Columns:
- `url`: The full website link
- `type`: Label (e.g. benign, phishing, defacement, malware)

Labels are encoded as:
- 0 = Safe (benign)
- 1 = Malicious (all others)

---

## How It Works

1. URLs are converted into feature vectors using TF-IDF
2. A Random Forest classifier is trained with class balancing
3. Model and vectorizer are saved using `pickle`
4. A Streamlit app loads them to predict new URLs

---

## How to Run

## activated the virtual enivroment 
.venv\Scripts\Activate

### Train the model:
```
python train_model.py
```

### Launch the app:
```
streamlit run app.py
```

---

## Sample URLs to Test

| URL | Expected Result |
|-----|------------------|
| `https://www.google.com` | ✅ Safe |
| `http://paypal.account-verify.login.com` | 🚨 Malicious |
| `http://bit.ly/fake-gift` | 🚨 Malicious |
| `https://chat.openai.com` | ✅ Safe |

---

## 👨‍💻 Author

**Sujit Sanjay Deshmukh**  
B.Tech CSE – IEP Microsoft Program  
Parul University  
Email: sujitdeshmukh123@gmail.com  

---

## License

This project is open for academic and personal learning purposes.