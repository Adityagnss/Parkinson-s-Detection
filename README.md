# Parkinson's-Detection

# Parkinson’s Disease Detection from Voice Recordings 🎙️🧠

A deep learning-based approach to detect early-stage Parkinson's Disease from voice recordings using a novel **ConvLSTM-Transformer** architecture. The model effectively captures short- and long-term speech features and outperforms traditional methods in accuracy and explainability.

---

## 🧪 Project Highlights

- 🚀 Designed a **ConvLSTM-Transformer** model tailored for temporal audio features.
- 🎯 Achieved **93.2% accuracy**, outperforming traditional RNN and CNN-based models.
- 🧠 Extracted clinically relevant features: **MFCCs**, **Jitter**, and **Shimmer**.
- 📈 Integrated **SHAP (SHapley Additive exPlanations)** for model interpretability.
- 📊 Developed a **continuous risk scoring system** to aid early diagnosis.

---

## 📁 Dataset

We used publicly available Parkinson’s Disease voice datasets, containing recordings of sustained phonations (like “ahh”) from both healthy individuals and Parkinson’s patients.

- Each sample was processed to extract:
  - **MFCCs (Mel-Frequency Cepstral Coefficients)**
  - **Jitter** – variation in frequency
  - **Shimmer** – variation in amplitude

> 📌 The data was preprocessed and normalized before feeding into the deep learning model.

---

## 🧠 Model Architecture

Our custom hybrid architecture includes:

### 🔸 ConvLSTM Block
- Captures short-term and localized temporal dependencies.
- Ideal for sequential data with local patterns.

### 🔹 Transformer Encoder
- Captures long-term dependencies using self-attention.
- Allows parallel processing and contextual understanding across frames.

### 🛠️ Explainability with SHAP
- Visualized feature importances to identify which parts of the audio influenced predictions.
- Improved trust and transparency in medical decision-making.

---

## 🧪 Performance

| Model               | Accuracy   |
|---------------------|------------|
| CNN                 | 88.4%      |
| LSTM                | 89.7%      |
| ConvLSTM-Transformer| **93.2%**  |

✅ Consistently outperformed baselines on key metrics like precision, recall, and F1-score.

---

## 🔍 Risk Scoring System

- Introduced a continuous **risk score (0 to 1)** for each prediction.
- Allows clinicians to assess the severity and monitor progression over time.

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/Adityagnss/Parkinson-s-Detection.git
cd Parkinson-s-Detection
