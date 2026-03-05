# Machine Learning for Antimicrobial Materials: A Comprehensive Tutorial

This repository contains a series of Jupyter notebooks designed to guide you through a complete machine learning workflow for predicting antimicrobial activity of materials. The project integrates real-world materials science data with state‑of‑the‑art data analysis and modeling techniques, providing a hands‑on learning experience for both beginners and experienced practitioners.

## Table of Contents

1. [Overview](#overview)
2. [Tutorial Structure](#tutorial-structure)
3. [Notebook Summaries](#notebook-summaries)
   - [Lessons 1–4: Foundation](#lessons-1-4-foundation)
   - [Notebook 2: Accessing Real DFT Data from the Materials Project](#notebook-2-accessing-real-dft-data-from-the-materials-project)
   - [Notebook 3: Feature Engineering & Exploratory Data Analysis (Revised for Scientific Rigour)](#notebook-3-feature-engineering--exploratory-data-analysis-revised-for-scientific-rigour)
   - [Notebook 4: Machine Learning Modeling of Antimicrobial Activity](#notebook-4-machine-learning-modeling-of-antimicrobial-activity)
4. [Dataset and Features](#dataset-and-features)
5. [Installation and Requirements](#installation-and-requirements)
6. [How to Use This Repository](#how-to-use-this-repository)
7. [Results Summary](#results-summary)
8. [Scientific Insights](#scientific-insights)
9. [License and Citation](#license-and-citation)

---

## Overview

Antimicrobial materials play a crucial role in healthcare, food packaging, and water treatment. Discovering new materials with high antimicrobial activity is traditionally expensive and time‑consuming. This tutorial demonstrates how machine learning can accelerate this discovery process by leveraging density functional theory (DFT) data from the [Materials Project](https://materialsproject.org/) and literature‑derived descriptors.

The project is structured as a progressive tutorial: starting with fundamental Python and data science skills, then moving through data acquisition, feature engineering, exploratory data analysis, and finally building and interpreting machine learning models. By the end of this series, you will have a complete pipeline that can be adapted to similar materials informatics problems.

## Tutorial Structure

The repository contains three introductory lessons (Lessons 1–3) and three main project notebooks (Notebooks 2, 3, 4). The notebooks are designed to be executed sequentially.

```
.
├── Lesson_1_Python_Basics.ipynb
├── Lesson_2_Data_Manipulation_with_Pandas.ipynb
├── Lesson_3_Data_Visualization_with_Matplotlib_Seaborn.ipynb
├── Notebook_01_Python_DataScience_Fundamentals.ipynb
├── Notebook_02_Accessing_DFT_Data_Materials_Project.ipynb
├── Notebook_03_Feature_Engineering_EDA.ipynb
├── Notebook_04_Machine_Learning_Modeling.ipynb
└── README.md
```

- **Lessons 1–3** – Cover essential Python programming, data handling with pandas, and visualization with Matplotlib and Seaborn. These lessons ensure you have the necessary foundation before tackling the main project.
- **Notebook 01** – Summarizes Lessons 1 to 3, earlier covered.
- **Notebook 02** – Demonstrates how to query the Materials Project API to obtain real DFT data for candidate materials (oxides, nitrides, etc.) and store it in a structured format.
- **Notebook 03** – Performs rigorous feature engineering based on the scientific literature, creates target labels for three antimicrobial mechanisms (ion release, photocatalytic activity, polarity), and explores the data through statistical summaries and visualizations.
- **Notebook 04** – Builds and evaluates classification and regression models to predict antimicrobial activity and potency. Feature importance analysis reveals which material properties drive each mechanism.

## Notebook Summaries

### Lessons 1–3: Foundation

These introductory notebooks are designed for learners with little to no prior experience in Python. They cover:

- **Lesson 1:** Basic Python syntax, data types, loops, functions, and working with NumPy arrays.
- **Lesson 2:** Introduction to pandas – Series and DataFrames, reading/writing data, filtering, grouping, and merging.
- **Lesson 3:** Data visualization fundamentals – creating line plots, scatter plots, bar charts, histograms, and heatmaps with Matplotlib and Seaborn.

Each lesson includes hands‑on exercises and mini‑projects to reinforce the concepts.

### Notebook 01: Summarizes all Lessons from Lesson 1 to 3

### Notebook 02: Accessing Real DFT Data from the Materials Project

**Objective:** Learn how to programmatically access the Materials Project database and retrieve DFT‑computed properties for materials relevant to antimicrobial activity.

**Key steps:**

1. **API setup:** Obtain an API key from Materials Project and configure the `mp-api` client.
2. **Data query:** Search for materials by chemical system (e.g., oxides, nitrides, carbides) and filter by stability criteria.
3. **Data retrieval:** Download properties such as formation energy, band gap, density, volume, and elemental information.
4. **Data storage:** Save the collected data into a CSV file for subsequent analysis.

**What you will learn:**

- Using REST APIs in Python.
- Efficiently querying large materials databases.
- Building a custom dataset for machine learning.

### Notebook 03: Feature Engineering & Exploratory Data Analysis (Revised for Scientific Rigour)

**Objective:** Transform raw DFT data into meaningful features and create scientifically‑grounded target labels. Explore the data to understand distributions, correlations, and potential biases.

**Key steps:**

1. **Feature engineering:** Compute composition‑based descriptors (average electronegativity, atomic mass, number of elements, etc.) and indicator features for antimicrobial metals (Ag, Cu, Zn, etc.).
2. **Target construction:** Derive binary labels for three antimicrobial mechanisms based on literature thresholds (e.g., band gap < 3 eV for photocatalytic activity) and a continuous potency score combining multiple descriptors.
3. **Exploratory analysis:** Visualize feature distributions, pairwise correlations, and the balance of target classes.

**What you will learn:**

- Domain‑informed feature creation.
- Handling missing data and avoiding data leakage.
- Data visualization techniques for high‑dimensional datasets.

### Notebook 04: Machine Learning Modeling of Antimicrobial Activity

**Objective:** Build and evaluate machine learning models to predict antimicrobial activity and potency. Interpret model outputs to gain scientific insights.

**Key steps:**

1. **Data preparation:** Split data into training and test sets, scale features using `StandardScaler`.
2. **Binary classification:** Train Logistic Regression, Random Forest, Gradient Boosting, and SVM classifiers for each of the four binary targets (`Ion_Release_Active`, `Photocatalytic_Active`, `Polarity_Active`, `Antimicrobial_Active`). Evaluate using accuracy, precision, recall, F1, and ROC‑AUC.
3. **Regression:** Predict the continuous `Potency_Score` using linear models and tree‑based regressors. Avoid leakage by excluding features used in target construction.
4. **Feature importance:** Extract and visualize feature importances from tree‑based models to identify key descriptors for each mechanism.
5. **Discussion:** Relate findings back to the underlying science – e.g., why formation energy and band gap dominate certain mechanisms.

**What you will learn:**

- End‑to‑end machine learning pipeline.
- Model selection and hyperparameter tuning (optional).
- Evaluation metrics for classification and regression.
- Interpreting model results in a scientific context.

## Dataset and Features

The final dataset used in Notebook 4 contains **55,200 materials** with the following features and targets:

| Feature | Description |
|---------|-------------|
| `Band_Gap_eV` | DFT‑computed band gap (eV) |
| `Formation_Energy_eV_atom` | Formation energy per atom (eV) |
| `Density_g_cm3` | Density (g/cm³) |
| `Volume_A3` | Unit cell volume (Å³) |
| `Num_Elements` | Number of distinct elements |
| `Avg_Atomic_Number` | Average atomic number |
| `Avg_Atomic_Mass` | Average atomic mass (g/mol) |
| `Avg_Electronegativity` | Average electronegativity (Pauling scale) |
| `Max_Electronegativity` | Maximum electronegativity |
| `Min_Electronegativity` | Minimum electronegativity |
| `Electronegativity_Range` | Range of electronegativity |
| `Geometric_Mean_EN` | Geometric mean of electronegativity |
| `Has_Antimicrobial_Metal` | Indicator: contains Ag, Cu, or Zn |
| `Num_Antimicrobial_Metals` | Count of antimicrobial metals |

| Target | Type | Description |
|--------|------|-------------|
| `Ion_Release_Active` | Binary | Predicted active via ion‑release mechanism |
| `Photocatalytic_Active` | Binary | Predicted active via photocatalytic mechanism |
| `Polarity_Active` | Binary | Predicted active via polarity mechanism |
| `Antimicrobial_Active` | Binary | Combined activity (any mechanism) |
| `Potency_Score` | Continuous | Weighted score (0–100) combining mechanisms |

## Installation and Requirements

To run the notebooks, you need a Python 3 environment with the following packages:

```
numpy
pandas
matplotlib
seaborn
scikit-learn
mp-api          (for Notebook 2)
jupyter
```

Install all dependencies using:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn mp-api jupyter
```

For Notebook 2, you must also obtain a free API key from [Materials Project](https://materialsproject.org/api). Follow the instructions on their website to register and generate a key.

## How to Use This Repository

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/antimicrobial-ml-tutorial.git
   cd antimicrobial-ml-tutorial
   ```

2. **Install the required packages** (see above).

3. **Launch Jupyter Notebook:**

   ```bash
   jupyter notebook
   ```

4. **Work through the notebooks in order:**

   - Start with **Lesson 1**, **Lesson 2**, and **Lesson 3** if you need to build foundational skills or go straight to Notebook 01.
   - Proceed to **Notebook 02** to fetch your own DFT data (or use the provided CSV if you prefer to skip this step).
   - Run **Notebook 03** to perform feature engineering and EDA.
   - Finally, execute **Notebook 04** to train and evaluate machine learning models.

5. **Experiment!** Modify model parameters, try different features, or adapt the code to your own materials science problems.

## Results Summary

Here are the key outcomes from Notebook 4:

- **Binary classification:** All models achieved excellent performance, with Random Forest and Gradient Boosting reaching accuracy >0.99 and F1‑scores >0.99 for most targets. This confirms that the engineered features are highly predictive.
- **Regression:** The Random Forest Regressor yielded the best results for `Potency_Score` (R² ≈ 0.85, MAE ≈ 2.29). Linear models underperformed (R² ≈ 0.65), indicating non‑linear relationships.
- **Feature importance:** The most important features overall were `Has_Antimicrobial_Metal`, `Formation_Energy_eV_atom`, and `Band_Gap_eV`. Different mechanisms are driven by different descriptors (e.g., photocatalytic activity depends strongly on band gap).

## Scientific Insights

- **Ion release** is associated with the presence of antimicrobial metals and low formation energy (easy ion leaching).
- **Photocatalytic activity** requires a moderate band gap (typically <3 eV) and is influenced by electronegativity.
- **Polarity** correlates with electronegativity range and geometric mean.
- The combined `Antimicrobial_Active` label integrates these effects, with `Has_Antimicrobial_Metal` and `Formation_Energy` as the dominant predictors.

These insights align with domain knowledge and demonstrate how machine learning can uncover interpretable patterns in materials science.

## License and Citation

This project is licensed under the MIT License – feel free to use, modify, and distribute the code with attribution.

If you use this tutorial in your research or teaching, please cite it as:

```
Ogungbemiro, F. (2025). Machine Learning for Antimicrobial Materials: A Comprehensive Tutorial.
GitHub repository: https://github.com/festusoladayoonline-debug/antimicrobial-ml-tutorial
```

For questions or feedback, please open an issue on GitHub or contact the author directly.

---

Happy learning, and enjoy discovering new antimicrobial materials with machine learning!
