import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture  
from sklearn.cluster import DBSCAN
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score,
                           precision_score, recall_score, f1_score,
                           confusion_matrix, classification_report,
                           adjusted_rand_score, silhouette_score)
from sklearn.decomposition import PCA
import json
import os
from collections import defaultdict

# Configuration
DATA_PATH = "dermatology (1).csv"  
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Data Loading 
def load_data(filename):
    """Loading Dataset with robust error handling"""
    try:
        # Try common delimiters with explicit error handling
        for delimiter in [',', ';', '\t']:
            try:
                df = pd.read_csv(filename, delimiter=delimiter)
                
                # Validate we got meaningful data
                if df.shape[1] > 1 and df.shape[0] > 0:
                    break
            except pd.errors.ParserError:
                continue
            except UnicodeDecodeError:
                # Try different encoding if needed
                try:
                    df = pd.read_csv(filename, delimiter=delimiter, encoding='latin1')
                    if df.shape[1] > 1 and df.shape[0] > 0:
                        break
                except:
                    continue
        else:
            raise ValueError(f"Failed to read {filename} with any standard delimiter (tried: ',', ';', '\\t')")
        
        # Verify expected column count
        if df.shape[1] != 35:
            raise ValueError(f"Expected 35 columns, found {df.shape[1]} in {filename}")
        
        # Clean and convert data
        df.replace('?', np.nan, inplace=True)
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='raise')
            except ValueError:
                raise ValueError(f"Non-numeric values found in column {col} that couldn't be converted")
        
        # Impute missing values with column median
        df = df.fillna(df.median(numeric_only=True))
        
        return df
    
    except Exception as e:
        raise ValueError(f"Error loading {filename}: {str(e)}")
    
# Data Preprocessing
def preprocess_data(df):
    """Split data into features and target, then scale"""
    X = df.iloc[:, :-1]  # All features (34 columns)
    y = df.iloc[:, -1]   # Class label (last column)
    
    # Define feature subsets
    clinical_features = list(range(0, 12))  # First 12 are clinical
    histo_features = list(range(12, 34))    # Next 22 are histopathology
    
    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'feature_subsets': {
            'all': list(range(34)),
            'clinical': clinical_features,
            'histo': histo_features
        }
    }

# Model 1: Gradient Descent Regression
class GradientDescentRegressor:
    """Manual implementation of gradient descent for linear regression"""
    
    def __init__(self, learning_rate=0.01, n_iter=1000):
        self.lr = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def fit(self, X, y):
        n_samples = X.shape[0]
        self.weights = 0
        self.bias = 0
        
        for i in range(self.n_iter):
            y_pred = self.weights * X + self.bias
            
            # Compute gradients
            dw = (1/n_samples) * np.sum(X * (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # Compute and store cost
            cost = (1/(2*n_samples)) * np.sum((y_pred - y)**2)
            self.cost_history.append(cost)
    
    def predict(self, X):
        return self.weights * X + self.bias

def run_gradient_descent(X_train, X_test, y_train, y_test):
    """Train and evaluate gradient descent regression with multiple seeds"""
    print("\nMODEL 1: GRADIENT DESCENT REGRESSION (AGE ONLY)")
    
    
    # Extract age feature (last column)
    X_train_age = X_train[:, -1]
    X_test_age = X_test[:, -1]
    
    # Run with multiple seeds
    seeds = [42, 123, 456, 789, 101112]
    results = []
    
    for seed in seeds:
        np.random.seed(seed)
        gd = GradientDescentRegressor(learning_rate=0.1, n_iter=1000)
        gd.fit(X_train_age, y_train)
        
        # Make predictions
        train_pred = gd.predict(X_train_age)
        test_pred = gd.predict(X_test_age)
        
        # Evaluate performance
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        results.append({
            'seed': seed,
            'weights': gd.weights,
            'bias': gd.bias,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'cost_history': gd.cost_history
        })
    
    # Calculate mean ± std metrics
    metrics = ['train_mse', 'test_mse', 'train_r2', 'test_r2']
    summary = {
        metric: {
            'mean': np.mean([r[metric] for r in results]),
            'std': np.std([r[metric] for r in results])
        } for metric in metrics
    }
    
    # Save detailed results and summary
    with open(f'{OUTPUT_DIR}/gd_results.json', 'w') as f:
        json.dump({
            'individual_runs': results,
            'summary_stats': summary
        }, f, indent=4)
    
    # Plot cost vs iterations (using first seed)
    plt.figure(figsize=(8, 5))
    plt.plot(results[0]['cost_history'])
    plt.title("Gradient Descent: Cost vs Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Cost (MSE)")
    plt.grid()
    plt.savefig(f'{OUTPUT_DIR}/gd_cost_vs_iter.png')
    plt.close()
    
    return {
        'model': 'Gradient Descent Regression',
        'summary_stats': summary
    }

# Model 2: Random Forest Classifier
def run_random_forest(X_train, X_test, y_train, y_test, feature_subsets):
    """Train and evaluate Random Forest classifier with feature subsets"""
    print("\nMODEL 2: RANDOM FOREST CLASSIFIER")
    
    # Define parameter grid for tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    
    # Initialize model
    rf = RandomForestClassifier(random_state=42)
    
    # Store results for each feature subset
    subset_results = {}
    
    for subset_name, features in feature_subsets.items():
        print(f"\nRunning on {subset_name} features...")
        X_train_sub = X_train[:, features]
        X_test_sub = X_test[:, features]
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, return_train_score=True)
        grid_search.fit(X_train_sub, y_train)
        
        # Save full grid search results
        cv_results = pd.DataFrame(grid_search.cv_results_)
        cv_results.to_csv(f'{OUTPUT_DIR}/rf_{subset_name}_grid_results.csv', index=False)
        
        # Stability runs with best params
        stability_accs = []
        best_params = grid_search.best_params_
        
        for seed in [42, 123, 456, 789, 101112]:
            rf_stable = RandomForestClassifier(**best_params, random_state=seed)
            rf_stable.fit(X_train_sub, y_train)
            test_acc = accuracy_score(y_test, rf_stable.predict(X_test_sub))
            stability_accs.append(test_acc)
        
        # Get best model
        best_rf = grid_search.best_estimator_
        test_pred = best_rf.predict(X_test_sub)
        
        # Evaluate performance
        test_acc = accuracy_score(y_test, test_pred)
        conf_matrix = confusion_matrix(y_test, test_pred)
        
        subset_results[subset_name] = {
            'best_params': best_params,
            'test_accuracy': test_acc,
            'stability_runs': {
                'mean': np.mean(stability_accs),
                'std': np.std(stability_accs),
                'values': stability_accs
            },
            'confusion_matrix': conf_matrix.tolist()
        }
        
        # Plot feature importance for full feature set
        if subset_name == 'all':
            plt.figure(figsize=(12, 6))
            importances = best_rf.feature_importances_
            plt.bar(range(len(importances)), importances)
            plt.title("Random Forest Feature Importances")
            plt.xlabel("Feature Index")
            plt.ylabel("Importance Score")
            plt.savefig(f'{OUTPUT_DIR}/rf_feature_importance.png')
            plt.close()
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            plt.imshow(conf_matrix, cmap='Blues')
            plt.title("Confusion Matrix")
            plt.colorbar()
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.savefig(f'{OUTPUT_DIR}/rf_confusion_matrix.png')
            plt.close()
    
    # Save subset results
    with open(f'{OUTPUT_DIR}/rf_subset_results.json', 'w') as f:
        json.dump(subset_results, f, indent=4)
    
    return {
        'model': 'Random Forest Classifier',
        'subset_results': subset_results
    }

# Model 3: kNN Classifier
def run_knn(X_train, X_test, y_train, y_test, feature_subsets):
    """Train and evaluate kNN classifier with feature subsets"""
    print("\nMODEL 3: kNN CLASSIFIER")
    print("="*60)
    
    k_values = range(1, 21)
    results = {}
    
    for subset_name, features in feature_subsets.items():
        print(f"\nRunning on {subset_name} features...")
        X_train_sub = X_train[:, features]
        
        # Cross-validation for k selection
        cv_scores = defaultdict(list)
        
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn, X_train_sub, y_train, cv=5, scoring='accuracy')
            cv_scores[k] = {
                'mean': np.mean(scores),
                'std': np.std(scores)
            }
        
        # Save CV results
        with open(f'{OUTPUT_DIR}/knn_{subset_name}_cv_results.json', 'w') as f:
            json.dump(cv_scores, f, indent=4)
        
        # Find best k
        best_k = max(cv_scores.items(), key=lambda x: x[1]['mean'])[0]
        
        # Stability runs with best k
        X_test_sub = X_test[:, features]
        stability_accs = []
        
        for seed in [42, 123, 456, 789, 101112]:
            knn = KNeighborsClassifier(n_neighbors=best_k)
            knn.fit(X_train_sub, y_train)
            test_acc = accuracy_score(y_test, knn.predict(X_test_sub))
            stability_accs.append(test_acc)
        
        results[subset_name] = {
            'best_k': best_k,
            'cv_results': cv_scores,
            'stability_runs': {
                'mean': np.mean(stability_accs),
                'std': np.std(stability_accs),
                'values': stability_accs
            }
        }
        
        # Plot k vs accuracy for full feature set
        if subset_name == 'all':
            plt.figure(figsize=(10, 5))
            plt.plot(k_values, [cv_scores[k]['mean'] for k in k_values], marker='o')
            plt.axvline(x=best_k, color='r', linestyle='--', label=f'Best k: {best_k}')
            plt.title("kNN: Accuracy vs k Value (5-fold CV)")
            plt.xlabel("k Value")
            plt.ylabel("Cross-Validated Accuracy")
            plt.legend()
            plt.grid()
            plt.savefig(f'{OUTPUT_DIR}/knn_k_vs_accuracy.png')
            plt.close()
            
            # Plot confusion matrix for full feature set
            knn = KNeighborsClassifier(n_neighbors=best_k)
            knn.fit(X_train_sub, y_train)
            test_pred = knn.predict(X_test_sub)
            conf_matrix = confusion_matrix(y_test, test_pred)
            
            plt.figure(figsize=(8, 6))
            plt.imshow(conf_matrix, cmap='Blues')
            plt.title("Confusion Matrix")
            plt.colorbar()
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.savefig(f'{OUTPUT_DIR}/knn_confusion_matrix.png')
            plt.close()
    
    # Save results
    with open(f'{OUTPUT_DIR}/knn_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    return {
        'model': 'kNN Classifier',
        'subset_results': results
    }

# Model 4: Gaussian Mixture Model Clustering (Replaced KMeans)
def run_gmm(X_scaled, y):
    """Train and evaluate Gaussian Mixture Model clustering with multiple seeds"""
    print("\nMODEL 4: GAUSSIAN MIXTURE MODEL CLUSTERING")
    
    n_components_range = range(2, 10)
    covariance_types = ['full', 'tied', 'diag', 'spherical']
    seeds = [42, 123, 456, 789, 101112]
    
    results = {
        k: {
            cov_type: {'ari': [], 'silhouette': [], 'bic': []}
            for cov_type in covariance_types
        }
        for k in n_components_range
    }
    
    for seed in seeds:
        for k in n_components_range:
            for cov_type in covariance_types:
                try:
                    gmm = GaussianMixture(
                        n_components=k,
                        covariance_type=cov_type,
                        random_state=seed
                    )
                    clusters = gmm.fit_predict(X_scaled)
                    
                    ari = adjusted_rand_score(y, clusters)
                    sil = silhouette_score(X_scaled, clusters)
                    bic = gmm.bic(X_scaled)
                    
                    results[k][cov_type]['ari'].append(ari)
                    results[k][cov_type]['silhouette'].append(sil)
                    results[k][cov_type]['bic'].append(bic)
                except:
                    continue
    
    # Calculate mean ± std metrics
    summary = {
        k: {
            cov_type: {
                'ari': {'mean': np.mean(results[k][cov_type]['ari']), 'std': np.std(results[k][cov_type]['ari'])},
                'silhouette': {'mean': np.mean(results[k][cov_type]['silhouette']), 'std': np.std(results[k][cov_type]['silhouette'])},
                'bic': {'mean': np.mean(results[k][cov_type]['bic']), 'std': np.std(results[k][cov_type]['bic'])}
            }
            for cov_type in covariance_types
        }
        for k in n_components_range
    }
    
    # Save results
    with open(f'{OUTPUT_DIR}/gmm_results.json', 'w') as f:
        json.dump({
            'full_results': results,
            'summary_stats': summary
        }, f, indent=4)
    
    # Find best configuration based on ARI
    best_config = None
    best_ari = -np.inf
    
    for k in n_components_range:
        for cov_type in covariance_types:
            current_ari = summary[k][cov_type]['ari']['mean']
            if current_ari > best_ari:
                best_ari = current_ari
                best_config = (k, cov_type)
    
    best_k, best_cov_type = best_config
    
    # Fit final model with best configuration
    final_gmm = GaussianMixture(
        n_components=best_k,
        covariance_type=best_cov_type,
        random_state=42
    )
    final_clusters = final_gmm.fit_predict(X_scaled)
    
    # Plot metrics
    plt.figure(figsize=(15, 5))
    
    # ARI plot
    plt.subplot(1, 3, 1)
    for cov_type in covariance_types:
        plt.errorbar(
            n_components_range,
            [summary[k][cov_type]['ari']['mean'] for k in n_components_range],
            yerr=[summary[k][cov_type]['ari']['std'] for k in n_components_range],
            label=f'{cov_type}',
            marker='o'
        )
    plt.title("GMM: ARI vs Number of Components")
    plt.xlabel("Number of Components")
    plt.ylabel("Adjusted Rand Index")
    plt.legend()
    plt.grid()
    
    # Silhouette plot
    plt.subplot(1, 3, 2)
    for cov_type in covariance_types:
        plt.errorbar(
            n_components_range,
            [summary[k][cov_type]['silhouette']['mean'] for k in n_components_range],
            yerr=[summary[k][cov_type]['silhouette']['std'] for k in n_components_range],
            label=f'{cov_type}',
            marker='o'
        )
    plt.title("GMM: Silhouette Score vs Components")
    plt.xlabel("Number of Components")
    plt.ylabel("Silhouette Score")
    plt.legend()
    plt.grid()
    
    # BIC plot
    plt.subplot(1, 3, 3)
    for cov_type in covariance_types:
        plt.errorbar(
            n_components_range,
            [summary[k][cov_type]['bic']['mean'] for k in n_components_range],
            yerr=[summary[k][cov_type]['bic']['std'] for k in n_components_range],
            label=f'{cov_type}',
            marker='o'
        )
    plt.title("GMM: BIC vs Number of Components")
    plt.xlabel("Number of Components")
    plt.ylabel("Bayesian Information Criterion")
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/gmm_scores.png')
    plt.close()
    
    # Visualize clusters with PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=final_clusters, cmap='viridis')
    plt.title(f"GMM Clusters (k={best_k}, cov={best_cov_type})")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(label="Cluster")
    plt.savefig(f'{OUTPUT_DIR}/gmm_pca.png')
    plt.close()
    
    return {
        'model': 'Gaussian Mixture Model',
        'best_k': best_k,
        'best_cov_type': best_cov_type,
        'summary_stats': summary
    }

# Model 5: DBSCAN Clustering
def run_dbscan(X_scaled, y):
    """Train and evaluate DBSCAN clustering with parameter grid search"""
    print("\nMODEL 5: DBSCAN CLUSTERING")
    
    
    # Parameter grid
    eps_values = np.linspace(0.1, 2.0, 10)
    min_samples_values = range(2, 10)
    seeds = [42, 123, 456, 789, 101112]
    
    results = []
    
    for seed in seeds:
        np.random.seed(seed)
        seed_results = []
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                clusters = dbscan.fit_predict(X_scaled)
                
                # Only compute metrics if valid clusters found
                n_clusters = len(np.unique(clusters))
                if n_clusters > 1:
                    ari = adjusted_rand_score(y, clusters)
                    sil = silhouette_score(X_scaled, clusters)
                else:
                    ari = -1
                    sil = -1
                
                seed_results.append({
                    'eps': eps,
                    'min_samples': min_samples,
                    'ari': ari,
                    'silhouette': sil,
                    'n_clusters': n_clusters
                })
        
        # Find best params for this seed
        valid_results = [r for r in seed_results if r['ari'] != -1]
        if valid_results:
            best_run = max(valid_results, key=lambda x: x['ari'])
            results.append({
                'seed': seed,
                'best_params': {'eps': best_run['eps'], 'min_samples': best_run['min_samples']},
                'best_ari': best_run['ari'],
                'best_silhouette': best_run['silhouette']
            })
    
    # Save results
    with open(f'{OUTPUT_DIR}/dbscan_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Run final model with median best parameters
    if results:
        median_eps = np.median([r['best_params']['eps'] for r in results])
        median_min_samples = int(np.median([r['best_params']['min_samples'] for r in results]))
        
        final_dbscan = DBSCAN(eps=median_eps, min_samples=median_min_samples)
        final_clusters = final_dbscan.fit_predict(X_scaled)
        
        # Visualize clusters with PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        plt.figure(figsize=(8, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=final_clusters, cmap='viridis')
        plt.title(f"DBSCAN Clusters (eps={median_eps:.2f}, min_samples={median_min_samples})")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.colorbar(label="Cluster")
        plt.savefig(f'{OUTPUT_DIR}/dbscan_pca.png')
        plt.close()
    
    return {
        'model': 'DBSCAN Clustering',
        'results': results
    }

# Main Execution
def main():
    # Load and preprocess data
    try:
        df = load_data(DATA_PATH)
        preprocessed = preprocess_data(df)
        
        # Scale full dataset for clustering
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df.iloc[:, :-1])
        y = df.iloc[:, -1].values
        
    except Exception as e:
        print(f"Error loading/preprocessing data: {e}")
        return
    
    # Run all models
    results = {}
    
    # Model 1: Gradient Descent
    results['model1'] = run_gradient_descent(
        preprocessed['X_train_scaled'],
        preprocessed['X_test_scaled'],
        preprocessed['y_train'],
        preprocessed['y_test']
    )
    
    # Model 2: Random Forest
    results['model2'] = run_random_forest(
        preprocessed['X_train_scaled'],
        preprocessed['X_test_scaled'],
        preprocessed['y_train'],
        preprocessed['y_test'],
        preprocessed['feature_subsets']
    )
    
    # Model 3: kNN
    results['model3'] = run_knn(
        preprocessed['X_train_scaled'],
        preprocessed['X_test_scaled'],
        preprocessed['y_train'],
        preprocessed['y_test'],
        preprocessed['feature_subsets']
    )
    
    # Model 4: Gaussian Mixture Model (Replaced KMeans)
    results['model4'] = run_gmm(X_scaled, y)
    
    # Model 5: DBSCAN Clustering
    results['model5'] = run_dbscan(X_scaled, y)
    
    # Save combined results
    with open(f'{OUTPUT_DIR}/combined_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print summary comparison
    print("\nMODEL COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Model':<30} {'Metric':<15} {'Value':<10}")
    print("-"*60)
    
    # Gradient Descent
    gd_stats = results['model1']['summary_stats']
    print(f"{'Gradient Descent':<30} {'Test R²':<15} {gd_stats['test_r2']['mean']:.4f} ± {gd_stats['test_r2']['std']:.4f}")
    
    # Random Forest (all features)
    rf_stats = results['model2']['subset_results']['all']
    print(f"{'Random Forest':<30} {'Test Accuracy':<15} {rf_stats['stability_runs']['mean']:.4f} ± {rf_stats['stability_runs']['std']:.4f}")
    
    # kNN (all features)
    knn_stats = results['model3']['subset_results']['all']
    print(f"{'kNN':<30} {'Test Accuracy':<15} {knn_stats['stability_runs']['mean']:.4f} ± {knn_stats['stability_runs']['std']:.4f}")
    
    # GMM
    gmm_stats = results['model4']['summary_stats']
    best_k = results['model4']['best_k']
    best_cov_type = results['model4']['best_cov_type']
    print(f"{'Gaussian Mixture':<30} {'Best ARI':<15} {gmm_stats[best_k][best_cov_type]['ari']['mean']:.4f} ± {gmm_stats[best_k][best_cov_type]['ari']['std']:.4f}")
    
    # DBSCAN
    if results['model5']['results']:
        dbscan_ari = np.mean([r['best_ari'] for r in results['model5']['results']])
        dbscan_std = np.std([r['best_ari'] for r in results['model5']['results']])
        print(f"{'DBSCAN':<30} {'Best ARI':<15} {dbscan_ari:.4f} ± {dbscan_std:.4f}")
    
    # Create comparison plot
    comparison_data = {
        'Gradient Descent (R²)': gd_stats['test_r2']['mean'],
        'Random Forest (Acc)': rf_stats['stability_runs']['mean'],
        'kNN (Acc)': knn_stats['stability_runs']['mean'],
        'GMM (ARI)': gmm_stats[best_k][best_cov_type]['ari']['mean'],
    }
    
    if results['model5']['results']:
        comparison_data['DBSCAN (ARI)'] = dbscan_ari
    
    plt.figure(figsize=(10, 6))
    plt.bar(comparison_data.keys(), comparison_data.values())
    plt.title("Model Performance Comparison")
    plt.ylabel("Performance Metric")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/model_comparison.png')
    plt.close()

if __name__ == "__main__":
    main()
    