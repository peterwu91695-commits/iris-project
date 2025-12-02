"""
Beginner-Friendly Machine Learning Project: Iris Flower Classification
========================================================================

This project demonstrates the complete ML workflow:
1. Load and explore data
2. Visualize data
3. Prepare data for training
4. Train multiple models
5. Evaluate and compare models
6. Make predictions
"""

# Step 1: Import Libraries
# -------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set random seed for reproducibility
np.random.seed(42)

# Create outputs directory if it doesn't exist
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configure plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


# Step 2: Load the Data
# ----------------------
print("=" * 60)
print("STEP 1: LOADING DATA")
print("=" * 60)

# Load the famous Iris dataset
iris = load_iris()
X = iris.data  # Features (measurements)
y = iris.target  # Target (species)

# Create a DataFrame for easier manipulation
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(y, iris.target_names)

print("\nDataset loaded successfully!")
print(f"Number of samples: {len(df)}")
print(f"Number of features: {len(iris.feature_names)}")
print(f"\nFirst 5 rows:")
print(df.head())


# Step 3: Explore the Data
# -------------------------
print("\n" + "=" * 60)
print("STEP 2: EXPLORING DATA")
print("=" * 60)

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nSpecies Distribution:")
print(df['species'].value_counts())

# Check for missing values
print(f"\nMissing values: {df.isnull().sum().sum()}")


# Step 4: Visualize the Data
# ---------------------------
print("\n" + "=" * 60)
print("STEP 3: VISUALIZING DATA")
print("=" * 60)

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Pairplot-style scatter for first two features
for species_name, species_group in df.groupby('species'):
    axes[0, 0].scatter(species_group['sepal length (cm)'], 
                       species_group['sepal width (cm)'],
                       label=species_name, alpha=0.6, s=50)
axes[0, 0].set_xlabel('Sepal Length (cm)')
axes[0, 0].set_ylabel('Sepal Width (cm)')
axes[0, 0].set_title('Sepal Length vs Width by Species')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Pairplot-style scatter for last two features
for species_name, species_group in df.groupby('species'):
    axes[0, 1].scatter(species_group['petal length (cm)'], 
                       species_group['petal width (cm)'],
                       label=species_name, alpha=0.6, s=50)
axes[0, 1].set_xlabel('Petal Length (cm)')
axes[0, 1].set_ylabel('Petal Width (cm)')
axes[0, 1].set_title('Petal Length vs Width by Species')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Box plot for petal length
df.boxplot(column='petal length (cm)', by='species', ax=axes[1, 0])
axes[1, 0].set_title('Petal Length Distribution by Species')
axes[1, 0].set_xlabel('Species')
axes[1, 0].set_ylabel('Petal Length (cm)')
plt.sca(axes[1, 0])
plt.xticks(rotation=45)

# 4. Correlation heatmap
correlation_matrix = df.iloc[:, :-1].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
            center=0, ax=axes[1, 1], fmt='.2f')
axes[1, 1].set_title('Feature Correlation Heatmap')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'iris_exploration.png'), dpi=300, bbox_inches='tight')
print(f"\n✓ Visualization saved as '{OUTPUT_DIR}/iris_exploration.png'")


# Step 5: Prepare Data for Training
# ----------------------------------
print("\n" + "=" * 60)
print("STEP 4: PREPARING DATA")
print("=" * 60)

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

# Feature scaling (normalize features to similar ranges)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n✓ Data scaled using StandardScaler")
print(f"Mean of scaled training data: {X_train_scaled.mean():.4f}")
print(f"Std of scaled training data: {X_train_scaled.std():.4f}")


# Step 6: Train Multiple Models
# ------------------------------
print("\n" + "=" * 60)
print("STEP 5: TRAINING MODELS")
print("=" * 60)

# Dictionary to store our models
models = {
    'Logistic Regression': LogisticRegression(max_iter=200, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
}

# Train each model and store results
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    
    # Calculate accuracy
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    results[name] = {
        'model': model,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'predictions': test_pred
    }
    
    print(f"  Training Accuracy: {train_acc:.4f}")
    print(f"  Testing Accuracy: {test_acc:.4f}")


# Step 7: Evaluate and Compare Models
# ------------------------------------
print("\n" + "=" * 60)
print("STEP 6: EVALUATING MODELS")
print("=" * 60)

# Find the best model
best_model_name = max(results, key=lambda x: results[x]['test_accuracy'])
best_model = results[best_model_name]['model']

print(f"\n🏆 Best Model: {best_model_name}")
print(f"Test Accuracy: {results[best_model_name]['test_accuracy']:.4f}")

# Detailed classification report for best model
print(f"\nDetailed Classification Report for {best_model_name}:")
print(classification_report(y_test, results[best_model_name]['predictions'],
                          target_names=iris.target_names))

# Confusion Matrix
cm = confusion_matrix(y_test, results[best_model_name]['predictions'])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
print(f"\n✓ Confusion matrix saved as '{OUTPUT_DIR}/confusion_matrix.png'")

# Model Comparison Chart
plt.figure(figsize=(10, 6))
model_names = list(results.keys())
train_accs = [results[m]['train_accuracy'] for m in model_names]
test_accs = [results[m]['test_accuracy'] for m in model_names]

x = np.arange(len(model_names))
width = 0.35

plt.bar(x - width/2, train_accs, width, label='Training Accuracy', alpha=0.8)
plt.bar(x + width/2, test_accs, width, label='Testing Accuracy', alpha=0.8)

plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Performance Comparison')
plt.xticks(x, model_names, rotation=45, ha='right')
plt.legend()
plt.ylim([0.9, 1.01])
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison.png'), dpi=300, bbox_inches='tight')
print(f"✓ Model comparison saved as '{OUTPUT_DIR}/model_comparison.png'")


# Step 8: Make Predictions on New Data
# -------------------------------------
print("\n" + "=" * 60)
print("STEP 7: MAKING PREDICTIONS")
print("=" * 60)

# Create example new flowers to classify
new_flowers = np.array([
    [5.1, 3.5, 1.4, 0.2],  # Should be Setosa
    [6.2, 2.9, 4.3, 1.3],  # Should be Versicolor
    [7.3, 3.0, 6.3, 1.8],  # Should be Virginica
])

# Scale the new data using the same scaler
new_flowers_scaled = scaler.transform(new_flowers)

# Make predictions
predictions = best_model.predict(new_flowers_scaled)
prediction_probs = best_model.predict_proba(new_flowers_scaled)

print("\nPredictions for new flower samples:")
print("-" * 60)
for i, (flower, pred, probs) in enumerate(zip(new_flowers, predictions, prediction_probs)):
    print(f"\nFlower {i+1}:")
    print(f"  Measurements: Sepal L={flower[0]}, Sepal W={flower[1]}, "
          f"Petal L={flower[2]}, Petal W={flower[3]}")
    print(f"  Predicted Species: {iris.target_names[pred]}")
    print(f"  Confidence:")
    for species, prob in zip(iris.target_names, probs):
        print(f"    {species}: {prob:.2%}")


# Step 9: Summary
# ---------------
print("\n" + "=" * 60)
print("PROJECT SUMMARY")
print("=" * 60)
print(f"""
✓ Loaded Iris dataset: {len(df)} samples, {len(iris.feature_names)} features
✓ Explored and visualized data patterns
✓ Split data: {len(X_train)} training, {len(X_test)} testing samples
✓ Trained {len(models)} different models
✓ Best model: {best_model_name} with {results[best_model_name]['test_accuracy']:.2%} accuracy
✓ Made predictions on new flower samples

Files created in '{OUTPUT_DIR}/' directory:
  - iris_exploration.png (data visualizations)
  - confusion_matrix.png (model evaluation)
  - model_comparison.png (performance comparison)
""")

print("\n🎉 Project completed successfully!")
print("=" * 60)