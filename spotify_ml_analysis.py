# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import time
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = "universal_top_spotify_songs2.csv"
data = pd.read_csv(file_path)

# Target variable creation
data['target'] = (data['popularity'] > 50).astype(int)

# Feature selection
features = [
    'danceability', 'energy', 'loudness', 'tempo', 'speechiness',
    'acousticness', 'instrumentalness', 'valence', 'duration_ms', 'is_explicit'
]

x = data[features]
y = data['target']

# Exploratory Data Analysis (EDA)
print("Dataset Information:")
print(data.info())
print("\nClass Distribution:")
print(data['target'].value_counts())
print("\nStatistical Summary:")
print(data.describe())

# Visualize feature relationships with the target
sns.boxplot(x='target', y='danceability', data=data)
plt.title("Danceability vs. Target")
plt.show()

# Data scaling
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
x_resampled, y_resampled = smote.fit_resample(x_scaled, y)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=42)

# Create a smaller training subset for computationally expensive models
x_train_small = x_train[:5000]
y_train_small = y_train[:5000]

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(kernel='linear'),  # Linear kernel for faster training
    'Random Forest': RandomForestClassifier(n_estimators=10, n_jobs=-1)  # Fewer estimators for faster training
}

# Train and evaluate models
results = {}
for name, model in models.items():
    start_time = time.time()

    # Train models with smaller dataset if needed
    if name in ['SVM', 'Random Forest']:
        model.fit(x_train_small, y_train_small)
    else:
        model.fit(x_train, y_train)

    # Make predictions and evaluate
    y_pred = model.predict(x_test)
    elapsed_time = time.time() - start_time
    results[name] = accuracy_score(y_test, y_pred)

    # Display results
    print(f"\nModel: {name}")
    print(f"Training Time: {elapsed_time:.2f} seconds")
    print(f"Accuracy: {results[name]:.4f}")
    print(classification_report(y_test, y_pred))

# Compare model performances
results_df = pd.DataFrame(results.items(), columns=['Model', 'Accuracy'])
print("\nModel Performances:")
print(results_df)

# Confusion Matrix Visualization
for name, model in models.items():
    y_pred = model.predict(x_test)
    disp = ConfusionMatrixDisplay.from_estimator(model, x_test, y_test)
    disp.ax_.set_title(f"{name} - Confusion Matrix")
    plt.show()

# ROC and AUC visualization
for name, model in models.items():
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(x_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Hyperparameter Tuning for Random Forest
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, scoring='accuracy')
grid_search.fit(x_train_small, y_train_small)
print("\nBest Parameters from Grid Search:")
print(grid_search.best_params_)
