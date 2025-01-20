import ray
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from ray.air import session
from ray import tune
from ray.tune.tuner import Tuner
from ray.air import RunConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.bayesopt import BayesOptSearch

# Initialize Ray (connect to the cluster)
ray.init(address="auto")

# Load the Digits dataset
def load_digits_data():
    digits = load_digits()
    X, y = digits.data, digits.target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Normalize features
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# SVM Training Function
def svm_train(config):
    X_train, X_val, y_train, y_val = load_digits_data()
    model = SVC(kernel=config.get("kernel","rbf"), C=config["C"], gamma=config.get("gamma", "scale"))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    session.report({"accuracy": accuracy})  # Report accuracy

storage_dir = "file:///home/cc/ray-cluster/reserve_kvm/.ray_results"

# Grid Search Tuning
def grid_search_tuning():
    grid_search_config = {
        "kernel": tune.grid_search(["linear", "rbf"]),
        "C": tune.grid_search([0.1, 1, 10]),
        "gamma": tune.grid_search([0.01, 0.1, 1])  # Only for rbf kernel
    }
    tuner = Tuner(
        svm_train,
        param_space=grid_search_config,
        tune_config=tune.TuneConfig(num_samples=1, metric="accuracy", mode="max"),
        run_config=RunConfig(name="grid_search_digits", storage_path=storage_dir)
    )
    results = tuner.fit()
    best_result = results.get_best_result(metric="accuracy", mode="max")
    print("Grid Search Best Hyperparameters:", best_result.config)

# Random Search Tuning
def random_search_tuning():
    random_search_config = {
        "kernel": tune.choice(["linear", "rbf"]),
        "C": tune.loguniform(0.1, 10),
        "gamma": tune.loguniform(0.01, 1)  # Only for rbf kernel
    }
    tuner = Tuner(
        svm_train,
        param_space=random_search_config,
        tune_config=tune.TuneConfig(num_samples=50, metric="accuracy", mode="max"),
        run_config=RunConfig(name="random_search_digits", storage_path=storage_dir)
    )
    results = tuner.fit()
    best_result = results.get_best_result(metric="accuracy", mode="max")
    print("Random Search Best Hyperparameters:", best_result.config)

# Bayesian Optimization Tuning
def bayesian_optimization_tuning():
    bayes_search_config = {
        "C": tune.loguniform(0.1, 10),
        "gamma": tune.loguniform(0.01, 1)  # Only for rbf kernel
    }
    bayesopt_search = BayesOptSearch(metric="accuracy", mode="max")
    tuner = Tuner(
        svm_train,
        param_space=bayes_search_config,
        tune_config=tune.TuneConfig(search_alg=bayesopt_search, num_samples=50),
        run_config=RunConfig(name="bayesopt_digits", storage_path=storage_dir)
    )
    results = tuner.fit()
    best_result = results.get_best_result(metric="accuracy", mode="max")
    print("Bayesian Optimization Best Hyperparameters:", best_result.config)

# Successive Halving Tuning (using ASHA Scheduler)
def successive_halving_tuning():
    asha_scheduler = ASHAScheduler(metric="accuracy", mode="max", grace_period=1, reduction_factor=2)
    asha_search_config = {
        "kernel": tune.choice(["linear", "rbf"]),
        "C": tune.loguniform(0.1, 10),
        "gamma": tune.loguniform(0.01, 1)  # Only for rbf kernel
    }
    tuner = Tuner(
        svm_train,
        param_space=asha_search_config,
        tune_config=tune.TuneConfig(scheduler=asha_scheduler, num_samples=50),
        run_config=RunConfig(name="asha_digits", storage_path=storage_dir)
    )
    results = tuner.fit()
    best_result = results.get_best_result(metric="accuracy", mode="max")
    print("Successive Halving Best Hyperparameters:", best_result.config)

# Run All Tuning Methods
if __name__ == "__main__":
    print("Running Grid Search...")
    grid_search_tuning()
    
    print("\nRunning Random Search...")
    random_search_tuning()
    
    print("\nRunning Bayesian Optimization...")
    bayesian_optimization_tuning()
    
    print("\nRunning Successive Halving...")
    successive_halving_tuning()
