import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load the CSV file
file_path = '/Users/andrianarisoniangola/desktop/andrianarison_project/pheno_table.csv'
pheno = pd.read_csv(file_path, converters={'features': lambda x: np.fromstring(x[1:-1], sep=' ')})

# Extract features and target variable
X = np.array(pheno['features'].tolist())  # Convert features to numpy array
y = pheno['adhd']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y, random_state=123)

# Initialize classifiers
classifiers = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "XGBoost": xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
}

# Calculate accuracy and F1 score for each classifier using cross-validation
confusion_matrices = []
results = {}
for clf_name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    conf_matrix = confusion_matrix(y_val, y_pred)

    confusion_matrices.append((clf_name, conf_matrix))

    results[clf_name] = {'Accuracy': accuracy, 'F1 Score': f1}
    print(f"{clf_name}: Accuracy={accuracy}, F1 Score={f1}")

# Create a grid layout for confusion matrices
num_classifiers = len(classifiers)
num_rows = int(np.ceil(num_classifiers / 3))  # Adjust number of columns as needed
fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows * 5))

# Plot and save confusion matrices in the grid
for i, (clf_name, conf_matrix) in enumerate(confusion_matrices):
    row = i // 3
    col = i % 3
    ax = axes[row, col]
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_title(f'Confusion Matrix - {clf_name}')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

# Adjust layout
plt.tight_layout()

# Save the entire grid as an image
plt.savefig('confusion_matrices.png')
plt.show()

# Create a DataFrame from results
results_df = pd.DataFrame(results).T

# Plot accuracy and F1 score for each model
results_df.plot(kind='bar', figsize=(10, 6))
plt.title('Classifier Performance Comparison')
plt.xlabel('Classifier')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.legend(loc='lower right')
plt.tight_layout()

# Save the graph to a new file
graph_file_path = '/Users/andrianarisoniangola/desktop/andrianarison_project/classifier_performance_comparison.png'
plt.savefig(graph_file_path)

plt.show()
