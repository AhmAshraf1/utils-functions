# utils-functions

A collection of reusable **Machine Learning** and **Deep Learning** utility functions for faster experimentation and model development.  
The repository is divided into three main modules:

- `utils_ml.py` → Classical ML utilities (data preprocessing, encoding, scaling, evaluation, visualization).  
- `utils_dl.py` → TensorFlow/Keras Deep Learning utilities (training visualization, inference time, dataset loaders).  
- `utils_dl_pytorch.py` → PyTorch Deep Learning utilities (training loops, validation, testing, visualization).  

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/AhmAshraf1/utils-functions.git
```

---

## 🔧 Example Usage

```Python
from utils_ml import encode_data, evaluate_classification_models
from utils_dl import measure_inference_time, plot_loss_accuracy
from utils_dl_pytorch import train_model, test_model
```

---

## 📚 Modules and Functions

🔹 `utils_ml.py` (Machine Learning Utilities)

### Data Cleaning & Preprocessing
- `analyze_IQR_outliers(data, num_columns)` – Detect outliers using IQR method
- `visualize_outliers(outlier_data, plot_type)` – Plot outlier counts/percentages
- `replace_outliers(data, column, value_to_replace)` – Replace outliers with a given value
- `visualize_nulls(data, plot_type)` – Show missing values
- `encode_data(X_train, X_test, encoder_type, columns)` – Encode categorical features
- `encode_target(y_train, y_test, encoder_type)` – Encode target labels
- `scale_data(X_train, X_test, scaler_type, columns)` – Scale features

### Model Evaluation
- `evaluate_regression_models(X_train, y_train, X_test, y_test, models)` -> Evaluate regression models → returns metrics + trained models.
- `evaluate_classification_models(X_train, y_train, X_test, y_test, models)` -> Evaluate classification models.
- `evaluate_classification_metrics(y_true, y_pred, target_names, display)` -> Generate confusion matrix + classification report.
- `evaluate_clustering_models(X_train, X_test, models)` -> Evaluate clustering models with silhouette score.
- `evaluate_models(X_train, y_train, X_test, y_test, models, task_type)` -> General wrapper for regression/classification.
- `get_voting(models_df, n_top, voting_type)` -> Build voting ensemble (classifier/regressor).

### Metrics & Visualization
- `accuracy_and_rmse(y_test, prediction)`
- `precision_recall_f1(y_test, prediction)`
- `plot_roc_auc_curve(y_test, y_prob)`
- `plot_precision_recall_curve(y_true, y_pred_proba)`

### Model Tuning
- `grid_search_classification_models(X, y)`
- `random_search_classification_models(X, y, n_iter)`

---

🔹 `utils_dl.py` (Deep Learning Utilities – TensorFlow/Keras)

### Training & Inference
- `measure_inference_time(model, input_data)` → Measure model inference time.
- `plot_loss_accuracy(history)` → Plot loss/accuracy curves.
- `learning_curves_plot(tr_data, start_epoch)`
- `learning_curves_tuning(tr_data, start_epoch, history_fine, fine_tune_epoch)`
- `plot_history(history)`

### Dataset Handling
- `train_val_test_data(batch_size, img_size, train_directory, validation_directory, test_directory)`
- `visualize_data(data)`
- `load_prep(img_path, img_title)`

### Predictions
- `random_image_predict(model, test_dir, data, rand_class, cls_name)`
- `predict_img(img_path, model, data)`

### Helpers
- `parse_image(filename)`
- `show(image, label)`

---

🔹 utils_dl_pytorch.py (Deep Learning Utilities – PyTorch)

### Reproducibility
- `set_seed(seed)` → Fix random seeds across libraries.

### Training & Validation
- `train_epoch(model, train_loader, criterion, optimizer, device)` → Train for one epoch.
- `validate_model(model, val_loader, criterion, device)` → Validate for one epoch.
- `train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, best_val_loss, best_val_acc, best_model_state)` -> Full training loop with history + checkpointing.

### Testing & Evaluation
- `test_model(model, test_loader, device)` → Classification report & confusion matrix.
- `visualize_results(model, test_loader, classes, num_images)` → Visualize test predictions.

### Visualization
- `learning_curves_tuning(history, fine_tune_epoch)` → Plot training curves.

### Utilities
- `EarlyStopping(patience, min_delta)` → Custom early stopping (class with .step(val_loss)).
- `CustomDataset(paths, transform, is_train)` → Dataset class with augmentation.
- `SimpleCNN(num_classes)` → Example CNN model.