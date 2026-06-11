# Iris Flower Classification 🌸

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-yellow)](https://pandas.pydata.org/)

> A comprehensive end-to-end machine learning pipeline for classifying Iris flower species using classical supervised learning algorithms.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project demonstrates the complete machine learning workflow—from exploratory data analysis to model evaluation—using the classic [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris). The goal is to classify Iris flowers into one of three species (*Iris setosa*, *Iris versicolor*, *Iris virginica*) based on four morphological features.

**Key Highlights:**
- Systematic comparison of 4 classification algorithms
- Hyperparameter tuning via `GridSearchCV`
- Cross-validation for robust performance estimation
- Feature importance analysis
- Interactive visualizations with Plotly

## Dataset

| Feature | Description |
|---------|-------------|
| `SepalLengthCm` | Length of the sepal in centimeters |
| `SepalWidthCm` | Width of the sepal in centimeters |
| `PetalLengthCm` | Length of the petal in centimeters |
| `PetalWidthCm` | Width of the petal in centimeters |
| `Species` | Target class: *Iris-setosa*, *Iris-versicolor*, *Iris-virginica* |

- **Samples:** 150 (50 per species)
- **Features:** 4 numeric
- **Classes:** 3 (balanced)

## Methodology

1. **Data Preprocessing**
   - Removed non-informative `Id` column
   - Label-encoded categorical target variable
   - 70/30 train-test split with `random_state=42` for reproducibility

2. **Model Selection**
   - Evaluated k-Nearest Neighbors, Decision Tree, Support Vector Machine, and Random Forest
   - 10-fold cross-validation for unbiased performance estimates

3. **Hyperparameter Tuning**
   - `GridSearchCV` on k-NN `n_neighbors` parameter (range: 1–30)

4. **Evaluation**
   - Accuracy, precision, recall, F1-score
   - Confusion matrix analysis
   - Feature importance (Random Forest)

## Results

### Model Comparison (10-Fold Cross-Validation)

| Model | CV Accuracy |
|-------|-------------|
| **k-NN (tuned, k=13)** | **98.00%** |
| SVM | 97.33% |
| k-NN (k=3) | 96.67% |
| Random Forest | 96.67% |
| Decision Tree | 96.00% |

### Test-Set Performance (k-NN k=3)

- **Accuracy:** 100.00%
- **Precision / Recall / F1:** 1.00 for all classes
- **Confusion Matrix:** Perfect classification (no misclassifications)

### Feature Importance

Random Forest analysis reveals that **petal dimensions** (length and width) are the most discriminative features for species classification, followed by sepal measurements.

## Installation

```bash
# Clone the repository
git clone https://github.com/VamshiKrishnaMacha/iris-flower-classification.git
cd iris-flower-classification

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run the Analysis Notebook

```bash
jupyter notebook notebooks/iris_classification.ipynb
```

### Run Tests

```bash
python -m pytest tests/
```

## Project Structure

```
iris-flower-classification/
├── data/
│   └── iris.csv              # Dataset
├── notebooks/
│   └── iris_classification.ipynb  # Main analysis notebook
├── src/
│   ├── __init__.py
│   ├── data_utils.py         # Data loading & preprocessing
│   ├── models.py            # Model training & tuning
│   └── evaluate.py          # Evaluation metrics & plots
├── tests/
│   └── test_models.py       # Unit tests
├── .github/
│   └── PULL_REQUEST_TEMPLATE.md
├── LICENSE
├── README.md
└── requirements.txt
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## License

This project is licensed under the MIT License—see the [LICENSE](LICENSE) file for details.

---

**Author:** Vamshi Krishna Macha

