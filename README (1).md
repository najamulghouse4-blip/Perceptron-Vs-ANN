# 🌸 Perceptron vs ANN — Iris Showdown

An interactive Streamlit app that pits a classic single-layer **Perceptron** against a small **Artificial Neural Network (ANN)** built in Keras/TensorFlow, trained head-to-head on the classic Iris flower dataset. Tune the hyperparameters, launch the training run, and watch live charts reveal the winner.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-app-ff4b4b)
![License](https://img.shields.io/badge/license-MIT-green)

## ✨ Features

- **Overview tab** — dataset stats, a gradient-styled preview table, and a species distribution donut chart
- **Explore tab** — interactive scatter plots (choose any two features), a correlation heatmap, and a 3D feature-space plot
- **Battle tab** — configure the experiment from the sidebar and launch a live training run:
  - Adjustable test split, random seed, ANN layer sizes, dropout rate, epochs, and batch size
  - Live progress bar while both models train
  - Winner callout with side-by-side accuracy cards
  - Accuracy comparison bar chart
  - ANN loss/accuracy training curve
  - Confusion matrix heatmaps for both models
  - Full classification reports (precision/recall/F1) for both models
- Custom dark, glassmorphism-styled UI with animated gradient title and Plotly-powered interactive charts throughout

## 📸 Preview

*(Add a screenshot or GIF of the app here once you've run it — drag an image into this section on GitHub.)*

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) — UI framework
- [scikit-learn](https://scikit-learn.org/) — Perceptron, preprocessing, train/test split, metrics
- [TensorFlow / Keras](https://www.tensorflow.org/) — ANN model
- [Plotly](https://plotly.com/python/) — interactive charts
- [pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data handling

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

## 📂 Project Structure

```
.
├── app.py             # Main Streamlit application
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## 🧠 How It Works

1. The **Iris dataset** (150 samples, 4 features, 3 species) is loaded via `sklearn.datasets.load_iris`.
2. Features are standardized with `StandardScaler`; labels are encoded with `LabelEncoder`.
3. The data is split into train/test sets (ratio configurable in the sidebar).
4. A `sklearn.linear_model.Perceptron` is trained on the raw split.
5. A small feed-forward ANN (`Dense → Dropout → Dense → Dropout → Dense(softmax)`) is trained on one-hot-encoded labels using categorical cross-entropy.
6. Both models are evaluated on the held-out test set, and results are visualized side by side.

## ⚙️ Configuration Options (Sidebar)

| Control | Description |
|---|---|
| Test set size | Fraction of data held out for testing |
| Random state | Seed for reproducibility |
| Layer 1 / Layer 2 neurons | Size of the ANN's hidden layers |
| Dropout rate | Dropout applied after each hidden layer |
| Epochs | Number of training epochs for the ANN |
| Batch size | Training batch size for the ANN |

## 📄 License

This project is licensed under the MIT License — feel free to use, modify, and share.

## 🙌 Acknowledgements

- Iris dataset: R.A. Fisher (1936), distributed via `scikit-learn`
- Built with Streamlit, scikit-learn, TensorFlow/Keras, and Plotly
