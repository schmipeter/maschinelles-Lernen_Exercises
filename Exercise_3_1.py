import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

# 1. Data Generation: create a synthetic classification dataset.
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                           n_redundant=5, random_state=6020)

# 2. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=6020
)

# 3. Simulate Partially Labeled Data (only 20% of training data is labeled)
n_total_train = len(X_train)
n_labeled = int(0.2 * n_total_train)
labeled_indices = np.random.choice(n_total_train, size=n_labeled, replace=False)
X_train_labeled = X_train[labeled_indices]
y_train_labeled = y_train[labeled_indices]

unlabeled_indices = np.setdiff1d(np.arange(n_total_train), labeled_indices)
X_train_unlabeled = X_train[unlabeled_indices]
# In a real setting, these labels are unknown; here we keep them for simulation.
y_train_unlabeled = y_train[unlabeled_indices]

print("Initial labeled samples:", len(y_train_labeled))
print("Initial unlabeled samples:", len(X_train_unlabeled))

results = {}  # To store test scores for each stage.

# 4. Initial Random Forest Training on the Partial Labeled Data
rf_initial = RandomForestClassifier(random_state=6020)
rf_initial.fit(X_train_labeled, y_train_labeled)
initial_score = rf_initial.score(X_test, y_test)
results['Initial'] = initial_score
print(f"Initial RF test score with {len(y_train_labeled)} labels: {initial_score:.4f}")

# 5. Hyperparameter Tuning with GridSearchCV on the Labeled Data
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_leaf': [1, 2, 5],
    'max_features': ['sqrt', 'log2']
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=6020),
                           param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_labeled, y_train_labeled)
best_rf = grid_search.best_estimator_
tuned_score = best_rf.score(X_test, y_test)
results['Tuned'] = tuned_score
print("Best parameters:", grid_search.best_params_)
print(f"Tuned RF test score: {tuned_score:.4f}")

# 6. Active Learning: select samples with the lowest maximum predicted probabilities.
probabilities = best_rf.predict_proba(X_train_unlabeled)
max_confidences = np.max(probabilities, axis=1)
n_active = 50  # number of samples to add during active learning
active_indices = np.argsort(max_confidences)[:n_active]
X_active = X_train_unlabeled[active_indices]
y_active = y_train_unlabeled[active_indices]  # In practice, these would be manually labeled.

# Augment the labeled set with active samples.
X_train_labeled = np.vstack([X_train_labeled, X_active])
y_train_labeled = np.concatenate([y_train_labeled, y_active])

# Remove the active samples from the unlabeled pool.
mask = np.ones(len(X_train_unlabeled), dtype=bool)
mask[active_indices] = False
X_train_unlabeled = X_train_unlabeled[mask]
y_train_unlabeled = y_train_unlabeled[mask]

rf_active = RandomForestClassifier(random_state=6020)
rf_active.fit(X_train_labeled, y_train_labeled)
active_learning_score = rf_active.score(X_test, y_test)
results['Active Learning'] = active_learning_score
print(f"RF test score after active learning ({len(y_train_labeled)} labels): {active_learning_score:.4f}")

# 7. Ensemble: build a VotingClassifier on the augmented data.
svc_model = SVC(kernel='rbf', probability=True, random_state=6020)
lr_model = LogisticRegression(max_iter=1000, random_state=6020)
voting_clf = VotingClassifier(
    estimators=[('rf', best_rf), ('svc', svc_model), ('lr', lr_model)],
    voting='soft'
)
voting_clf.fit(X_train_labeled, y_train_labeled)
ensemble_score = voting_clf.score(X_test, y_test)
results['Ensemble'] = ensemble_score
print(f"Voting Ensemble test score: {ensemble_score:.4f}")

# 8. Optional: Clustering to add representative samples from the remaining unlabeled data.
if len(X_train_unlabeled) > 0:
    n_clusters = min(10, len(X_train_unlabeled))
    kmeans = KMeans(n_clusters=n_clusters, random_state=6020)
    clusters = kmeans.fit_predict(X_train_unlabeled)
    
    representative_indices = []
    for cluster in range(n_clusters):
        cluster_indices = np.where(clusters == cluster)[0]
        if len(cluster_indices) > 0:
            cluster_data = X_train_unlabeled[cluster_indices]
            center = kmeans.cluster_centers_[cluster].reshape(1, -1)
            closest, _ = pairwise_distances_argmin_min(center, cluster_data)
            representative_indices.append(cluster_indices[closest[0]])
            
    if representative_indices:
        X_repr = X_train_unlabeled[representative_indices]
        y_repr = y_train_unlabeled[representative_indices]
        # Add representative samples to the labeled set.
        X_train_labeled = np.vstack([X_train_labeled, X_repr])
        y_train_labeled = np.concatenate([y_train_labeled, y_repr])
        
        # Retrain the ensemble on the further augmented data.
        voting_clf.fit(X_train_labeled, y_train_labeled)
        clustering_score = voting_clf.score(X_test, y_test)
        results['Clustering'] = clustering_score
        print(f"Voting Ensemble test score after clustering samples: {clustering_score:.4f}")

# 9. Plot an Overview of All Results
labels = list(results.keys())
scores = [results[label] for label in labels]

plt.figure(figsize=(8, 6))
bars = plt.bar(labels, scores, color='skyblue')
plt.ylim(0, 1)
plt.xlabel("Method")
plt.ylabel("Test Accuracy")
plt.title("Overview of Test Scores at Different Stages")

# Annotate each bar with the score.
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.2f}",
             ha='center', va='bottom')
plt.show()