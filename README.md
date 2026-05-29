# Analyzing Acoustic Features for Speech Emotion Classification

**Speech Emotion Recognition using the RAVDESS Corpus**

This repository contains the code for *Analyzing Acoustic Features for Speech Emotion Classification: A Comparative Study on the RAVDESS Male Corpus*, published in the National High School Journal of Science 2026.
[Paper Here](https://nhsjs.com/2026/analyzing-acoustic-features-for-speech-emotion-classification-a-comparative-study-on-the-ravdess-male-corpus/)

---

## Overview

This project investigates how well different machine learning models can detect emotion from speech, and which audio features and representations are most useful for doing so. All experiments use the male subset of the [RAVDESS](https://zenodo.org/record/1188976) dataset (12 actors, 8 emotion classes).

The experimental pipeline is organized into three phases, designed so that each phase answers one core question, and later phases build on the best configuration found in earlier ones. This keeps the total number of experiments manageable while still covering a wide range of models and conditions.

---

## Experimental Structure

### Phase 1: Representation Experiments


This phase uses only 1–2 fast, stable models (Logistic Regression and Random Forest) to isolate the effect of *how the audio is represented*, before introducing model complexity. The goal is to lock in the best temporal segmentation and feature set before running the full model comparison.

#### Test 1: Temporal Representation

Determines the best way to segment audio clips in time. The same model (Logistic Regression) is run under four segmentation strategies:

| Condition | Description |
|---|---|
| Sentence-level | Full original clips (~4–6 seconds each) |
| 4s windows | Audio split into 4-second segments |
| 2s windows | Audio split into 2-second segments with overlap |
| 1s windows | Audio split into 1-second segments with overlap |

Shorter windows capture more localized emotional cues, but if they're too short, there may not be enough acoustic context for the model to work with.

**Output per condition:** Mean accuracy, std accuracy, mean F1, std F1, precision, recall (across all LOOCV folds and total), plus an 8×8 confusion matrix aggregated across folds.

#### Test 2: Feature Group Importance

Using the best segmentation from Test 1, this test determines which *groups* of acoustic features matter most. Logistic Regression is run four times, each time using only one feature group (or all of them):

| Condition | Features included |
|---|---|
| MFCC only | 13 Mel-frequency cepstral coefficients (mean + std = 26 features) |
| Prosody only | Pitch statistics, energy (RMS) statistics, zero-crossing rate |
| Voice quality only | Jitter, shimmer, harmonic-to-noise ratio (HNR) |
| All features | Full 41-dimensional feature vector |

This reveals whether emotion is best captured by the shape of the sound spectrum (MFCCs), the rhythm and loudness of speech (prosody), or the stability and texture of the voice (voice quality).

**Output per condition:** Mean accuracy, std accuracy, mean F1, std F1.

#### Test 3: Feature Importance Ranking

A single run using Random Forest with the best segmentation and all features. Random Forest naturally provides feature importance scores (based on how much each feature reduces classification error across its decision trees), which produces a ranked list of the individual features that matter most for emotion detection.

**Output:** Feature importance scores, sorted in descending order.

---

### Phase 2: Model Comparison

The best segmentation and feature set from Phase 1 are locked in, and every model is evaluated under the same LOOCV pipeline for a fair comparison.

#### Classical Models

These are traditional machine learning models that operate on the extracted feature vectors directly. They are fast to train, interpretable, and serve as strong baselines.

| Model | Description |
|---|---|
| Logistic Regression | Linear classifier that models class probabilities directly. L2 regularization prevents overfitting. |
| Ridge Regression | Linear model with L2 penalty — similar to logistic regression but formulated as a regression problem. |
| Lasso Regression | Linear model with L1 penalty — tends to zero out less important features, acting as built-in feature selection. |
| Random Forest | Ensemble of decision trees that vote on the final prediction. Captures nonlinear patterns. |
| Gradient Boosting | Sequentially builds trees where each one corrects the errors of the previous. Often very accurate. |
| Support Vector Classifier (SVC) | Finds the optimal boundary (hyperplane) between emotion classes in feature space. |

#### Shallow Neural Network Models

These are multilayer perceptrons (MLPs), fully connected neural networks with varying depth and width. They test whether adding moderate complexity on top of the same features improves performance.

| Model | Description |
|---|---|
| 1-Layer MLP | Single hidden layer. The simplest possible neural network beyond a linear model. |
| 3-Layer MLP | Three hidden layers with dropout regularization. Tests whether deeper networks learn better representations. |
| Wide MLP | Fewer layers but more neurons per layer. Tests whether width matters more than depth. |

#### Deep Learning Models

These architectures attempt to learn more complex patterns from the feature sequences, treating the feature vector as a pseudo-temporal signal.

| Model | Description |
|---|---|
| 1D CNN | Applies convolutional filters across the feature dimension to detect local patterns. |
| CNN + LSTM | Combines convolutional layers (local pattern detection) with LSTM units (modeling longer-range dependencies) and an attention mechanism. |

**Output per model:** Mean accuracy, std accuracy, mean F1, std F1, precision, recall, confusion matrix. For neural models only: training loss, validation loss, training accuracy, and validation accuracy per epoch (to diagnose overfitting). Also recorded: number of parameters and training time.

All models are computed within each LOOCV fold, so the data loading and splitting only happens once per fold rather than once per model.

---

### Phase 3: Focused Analysis

The top 3 models from Phase 2 (based on accuracy and F1) are selected for deeper investigation.

#### Test 4: Emotional Intensity

RAVDESS includes both low-intensity and high-intensity versions of each emotion (except Neutral). This test runs the top 3 models separately on each intensity level to see whether stronger emotional expression makes classification easier.

**Output per model per intensity:** Accuracy, F1, confusion matrix.

#### Test 5: Final Representation Confirmation

A final sanity check that revisits the representation question from Phase 1, but now using the best-performing models from Phase 2 instead of just Logistic Regression. This confirms (or challenges) whether the optimal segmentation holds across model types.

Conditions: sentence-level vs. word-level vs. windowed segmentation, tested with the best classical model and the best neural model.

**Output per condition:** Accuracy, F1, confusion matrix.

---

## Evaluation

All experiments use **Leave-One-Speaker-Out Cross-Validation (LOOCV)**. In each fold, one of the 12 actors is held out as the test set, and the model trains on the remaining 11. This simulates a real-world scenario where the model encounters a speaker it has never heard before, preventing it from memorizing speaker-specific quirks instead of learning generalizable emotion cues.

Feature standardization (zero mean, unit variance) is always computed from the training fold only and then applied to the test fold, preventing data leakage.

Windowed segmentation is performed after the actor-level split, so overlapping windows from the same original clip never appear in both training and test sets.

---

## Features

The full feature vector contains 41 dimensions:

| Group | Features | Count |
|---|---|---|
| MFCCs | Mean and std of 13 Mel-frequency cepstral coefficients | 26 |
| Prosody | Pitch (mean, std, min, max, range), RMS energy (mean, std, min, max, range), zero-crossing rate (mean, std) | 13 |
| Voice Quality | Jitter, shimmer, harmonic-to-noise ratio (HNR) | 3 |

---

## Dataset

**RAVDESS** — Ryerson Audio-Visual Database of Emotional Speech and Song

This study uses the **male speech subset** (12 actors). Each actor recorded two sentences across 8 emotions (Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised), with low- and high-intensity variants for all emotions except Neutral.

The dataset is publicly available at: https://zenodo.org/record/1188976
