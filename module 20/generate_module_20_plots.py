import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons, make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.inspection import DecisionBoundaryDisplay
import os

# Set style for professional look
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 12

# Create images directory
image_dir = 'images'
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

def plot_bias_variance_tradeoff():
    """Generates a theoretical Bias-Variance Tradeoff plot."""
    x = np.linspace(0, 10, 100)
    variance = (x/10)**2
    bias_sq = ((10-x)/10)**2
    irreducible_error = 0.1 * np.ones_like(x)
    total_error = variance + bias_sq + irreducible_error
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, total_error, label='Total Error', color='black', linewidth=3)
    plt.plot(x, bias_sq, label='Bias²', color='blue', linestyle='--')
    plt.plot(x, variance, label='Variance', color='red', linestyle='--')
    plt.plot(x, irreducible_error, label='Irreducible Error', color='green', linestyle=':')
    
    plt.xlabel('Model Complexity', fontsize=14)
    plt.ylabel('Error', fontsize=14)
    plt.title('Bias-Variance Tradeoff', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Annotations for Optimal Model
    optimal_idx = np.argmin(total_error)
    plt.annotate('Optimal Model Complexity', 
                 xy=(x[optimal_idx], total_error[optimal_idx]), 
                 xytext=(x[optimal_idx], total_error[optimal_idx] + 0.5),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=12, ha='center')
    
    # Annotations for Underfitting/Overfitting
    plt.text(1, 0.8, 'High Bias\n(Underfitting)', ha='center', color='blue', fontsize=12)
    plt.text(9, 0.8, 'High Variance\n(Overfitting)', ha='center', color='red', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, 'bias_variance_tradeoff.png'), dpi=300)
    plt.close()
    print("Generated bias_variance_tradeoff.png")

def plot_ensemble_boundaries():
    """Compares decision boundaries of DT, RF, and AdaBoost."""
    X, y = make_moons(n_samples=200, noise=0.3, random_state=42)
    
    classifiers = [
        ("Decision Tree", DecisionTreeClassifier(max_depth=5, random_state=42)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("AdaBoost", AdaBoostClassifier(n_estimators=50, random_state=42))
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, (name, clf) in zip(axes, classifiers):
        clf.fit(X, y)
        DecisionBoundaryDisplay.from_estimator(
            clf, X, cmap=plt.cm.RdBu, alpha=0.6, ax=ax, response_method="predict"
        )
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu_r, edgecolors='k', s=40)
        ax.set_title(name, fontsize=14)
        ax.axis('off')
        
    plt.suptitle("Decision Boundaries: Single Tree vs. Ensembles", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, 'ensemble_boundaries.png'), dpi=300)
    plt.close()
    print("Generated ensemble_boundaries.png")

def plot_boosting_iterations():
    """Visualizes how AdaBoost updates weights over iterations."""
    # Create simple dataset
    X, y = make_classification(n_samples=20, n_features=2, n_informative=2, 
                               n_redundant=0, n_repeated=0, n_classes=2, 
                               n_clusters_per_class=1, random_state=42)
    
    # AdaBoost manually to expose sample weights (or use base estimator inside loop)
    # For visualization, we'll fit a stump, see misclassified and re-weight manually for clarity
    
    # Just fit separate stumps to simulate iterations
    # Iteration 1: Normal weights
    # Iteration 2: Increase weights of mislcassified
    # Iteration 3: Further increase
    
    # Actually, simpler to just fit AdaBoost and visualize its estimators? 
    # But estimators in sklearn are trained on weighted data.
    
    clf = AdaBoostClassifier(n_estimators=3, random_state=42)
    clf.fit(X, y)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # We need to simulate the sample weights for visualization because sklearn doesn't store them all easily accessible per stage in a way that matches distinct plots perfectly without some work.
    # Instead, let's just plot the 3 estimators and the final aggregate.
    # Or, better: Plot the decision boundary of the first 3 weak learners.
    
    for i, (ax, tree) in enumerate(zip(axes, clf.estimators_)):
        # Plot decision boundary of this specific tree
        DecisionBoundaryDisplay.from_estimator(
            tree, X, cmap=plt.cm.RdBu, alpha=0.4, ax=ax, response_method="predict"
        )
        # Plot data points
        # To show "weights", we could use size, but we don't have the exact weights used for *this* step easily without re-implementing.
        # Let's just show the points and the boundary to show how they focus on different parts.
        
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu_r, edgecolors='k', s=60)
        ax.set_title(f"Weak Learner {i+1}", fontsize=14)
        ax.axis('off')
    
    plt.suptitle("AdaBoost: Sequential Weak Learners Focusing on Errors", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, 'boosting_iterations.png'), dpi=300)
    plt.close()
    print("Generated boosting_iterations.png")

if __name__ == "__main__":
    print("Generating visualizations for Module 20...")
    plot_bias_variance_tradeoff()
    plot_ensemble_boundaries()
    plot_boosting_iterations()
    print("All visualizations generated successfully.")
