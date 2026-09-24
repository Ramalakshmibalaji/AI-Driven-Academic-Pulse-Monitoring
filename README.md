# AI-Driven Academic Pulse Monitoring Through Multi-Modal Learning Feature Fusion

## Project Overview

This project presents an AI-driven academic pulse monitoring approach for analyzing student learning behaviour and predicting academic performance using Learning Management System (LMS) data.

The system integrates multiple learning signals, including LMS engagement, assessment and learning activities, study behaviour, and weekly temporal learning patterns.

The overall workflow includes:

- LMS data extraction
- Data preprocessing
- Exploratory data analysis
- Feature engineering
- Feature selection
- Multi-modal feature fusion
- Temporal learning analysis
- Model development
- Model evaluation
- Student-wise performance prediction

## Dataset

The publication dataset contains anonymized course-level learning behaviour data.

- Records: 4,708
- Features: 50
- Course identifiers: anonymized as Course 1, Course 2, etc.
- Missing values: 0

Raw LMS data and personally identifiable student information are not included in this repository.

## Feature Categories

The dataset contains features related to:

- LMS engagement
- Course access and completion
- Programming and exercise activity
- Quiz activity
- Assessment performance
- Study behaviour
- Weekly temporal engagement
- Learning consistency

## Machine Learning and Deep Learning

The project includes implementations related to:

- Artificial Neural Network (ANN)
- Multi-Modal LSTM
- Hybrid LSTM-Transformer
- Feature selection
- Student-wise performance prediction

The Hybrid LSTM-Transformer model combines static behavioural features with weekly temporal learning activity to capture both behavioural and sequential learning patterns.

## Repository Contents

### Data Processing

- `Preprocessing.py` – Data preprocessing
- `Eda_Processess.py` – Exploratory data analysis
- `create_publication_dataset.py` – Publication dataset preparation
- `anonymize_course_names.py` – Course-name anonymization

### Feature Processing

- `feature_engineering.py` – Feature engineering
- `feature_selection.py` – Feature selection

### Model Development

- `ann_model.py` – ANN model
- `lstm.py` – LSTM model
- `hybrid_lstm_transformer.py` – Hybrid LSTM-Transformer model
- `model_comparison.py` – Model comparison
- `final_evaluation.py` – Model evaluation

### Prediction

- `Hybrid_Student_Prediction.py`
- `Student_Prediction.py`

### Publication Figures

- `Figure_3_Correlation_Heatmap.png`
- `Figure_4_Weekly_Student_Engagement.png`

## Data Privacy

The repository does not contain raw LMS extraction data or personally identifiable student information.

Course names in the publication dataset have been anonymized to generic identifiers such as `Course 1`, `Course 2`, and so on.

## Authors

**Nishalini B**  
**Rishi Lingam M**  
**Sri Sai Priya S**

## Project Title

**AI-DRIVEN ACADEMIC PULSE MONITORING THROUGH MULTI-MODAL LEARNING FEATURE FUSION**