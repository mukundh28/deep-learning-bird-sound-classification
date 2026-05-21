# 🐦 Bird Sound Classification using 1D CNN

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end deep learning system built for automatic bird species recognition using audio recordings. The project leverages MFCC-based audio feature extraction and a sequential 1D Convolutional Neural Network (Conv1D) architecture to classify 114 different bird species efficiently.

---

# 📌 Project Overview

Bird sound recognition is an important task in ecological monitoring, biodiversity conservation, and wildlife research. Manual identification of bird species from large-scale audio recordings is time-consuming and requires domain expertise.

This project automates the classification process using:
- **MFCC Feature Extraction:** Audio feature reduction matching human auditory perception patterns.
- **Sequential 1D Convolutional Neural Networks:** Slides kernels natively along the time-series dimension to handle sequence constraints efficiently.
- **Data Cleaning Pipelines:** Integrated try-except engineering constraints to safely filter out corrupted metadata arrays or 0 KB elements without crashing.
- **Streamlit-based Real-Time Inference:** A responsive web application utilizing cached model instances to deliver immediate predictions.

The system is capable of classifying **114 unique bird species** directly from `.mp3` and `.wav` audio recordings.

---

# ⚙️ Processing Pipeline

```text
[Raw Audio (.mp3/.wav)]
         ↓
[Librosa Audio Loading]
         ↓
[40 MFCC Sequential Feature Extraction]
         ↓
[Padding / Truncation]
         ↓
[1D Convolutional Neural Network]
         ↓
[Softmax Classification]
         ↓
[Predicted Bird Species]
```

### 🎵 Acoustic Feature Engineering
Raw time-domain audio waves contain immense sample rates that are computationally heavy for neural networks to interpret directly.
* **Extraction:** The pipeline uses `librosa` to compute **40 Mel Frequency Cepstral Coefficients (MFCCs)**, transforming raw signals into low-dimensional acoustic "fingerprints" representing frequency changes over time.
* **Temporal Pooling:** Frame arrays are mean-reduced along the temporal sequence axis (`np.mean(axis=1)`) to compute a uniform, stable descriptor representing the entire duration of the clip.
* **Tensor Transformations:** Extracted vectors are packed as TensorFlow tensors and expanded along the final channel axis to scale input structures cleanly into shapes of `(None, 40, 1)`, providing the exact matching format required by the 1D-CNN input boundaries.

---

# 🧠 Model Architecture (1D-CNN)

Instead of converting audio data into heavy 2D spectrogram images and using heavy resource-intensive computer vision architectures, this repository implements a specialized **1D Convolutional Neural Network**. Because audio feature frames represent sequential patterns evolving step-by-step, the 1D kernels slide strictly along the temporal scale, capturing structural signal traits much faster and using significantly less computational memory.

* **Total Parameters:** 1,144,434 (~4.37 MB)
* **Trainable Parameters:** 1,143,154 (~4.36 MB)
* **Non-trainable Parameters:** 1,280 (~5.00 KB)

---

# 📊 Performance Analysis & Results

The system was optimized using the **Adam Optimizer** (learning rate: `1e-4`). To prevent overfitting and capture the optimal weight state, **Early Stopping** was implemented with a patience of 15 epochs over a maximum threshold of **150 epochs**. 

### Performance Metrics
We utilize standard classification metrics to gauge performance:

* **Categorical Accuracy:** Computes how frequently predictions match the ground-truth targets.
  $$Accuracy = \frac{Correct Predictions}{Total Predictions}$$

* **Sparse Categorical Crossentropy Loss:** Monitors predictive certainty penalization.
  $$\mathcal{L} = -\sum_{i} y_i \log(\hat{y}_i)$$

| Metric | Score |
|---|---|
| Final Training Accuracy | **99.59%** |
| Final Validation Accuracy | **98.44%** |
| Test Accuracy (Unseen Data) | **80.86%** |

*The high convergence metrics on training models highlight exceptional representation capability. While the delta observed on completely unseen partitions suggests a minor degree of variance overfitting due to ambient real-world recording dynamics, a solid 80%+ cross-validation success rate over 114 distinct species proves resilient real-world classification boundaries.*

---

# 📈 Training Analysis

### Training vs Validation Accuracy & Loss
![Training Accuracy](images/training-graph.png)

### Confusion Matrix
![Confusion Matrix](images/confusion-matrix.png)

---

# 💻 Streamlit Deployment

The local inference module is bundled as an interactive **Streamlit web application** that processes custom audio files on-the-fly and serves predictions instantly by utilizing cached `.keras` model states.

## Streamlit Interface
![Streamlit UI](images/streamlit-ui.png)

## Prediction Example
![Prediction Example](images/prediction-example.png)

---

# ▶️ Running the Project

### 🛠️ Local Environment Setup
1. Clone this repository to your local drive:
   ```bash
   git clone [https://github.com/mukundh28/deep-learning-bird-sound-classification.git](https://github.com/mukundh28/deep-learning-bird-sound-classification.git)
   cd deep-learning-bird-sound-classification
   ```
2. Install the necessary system dependencies:
   ```bash
   pip install tensorflow scikit-learn librosa numpy pandas matplotlib streamlit streamlit_extras opencv-python tqdm IPython
   ```
3. Boot up the user dashboard locally:
   ```bash
   streamlit run app.py
   ```

---

# 👤 Author

**Mukundh Reddy**


# 📜 License
This project is open-source and licensed under the **MIT License**.
