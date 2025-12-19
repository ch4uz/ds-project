from math import sqrt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import numpy as np
from numpy import array, ndarray, argsort, arange, std
from copy import deepcopy
from matplotlib.pyplot import subplots
from matplotlib.pyplot import figure, savefig, show, subplots
from pandas import DataFrame, concat, Series
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, plot_tree
from typing import Literal

from dslabs_functions import \
    CLASS_EVAL_METRICS, DELTA_IMPROVE, plot_bar_chart, plot_multiline_chart, \
    HEIGHT, run_NB, run_KNN, evaluate_approach, plot_multibar_chart, plot_line_chart, \
    plot_confusion_matrix, plot_evaluation_results, plot_horizontal_bar_chart, \
    FORECAST_MEASURES, PAST_COLOR, FUTURE_COLOR, PRED_PAST_COLOR, \
    PRED_FUTURE_COLOR, set_chart_labels, DecomposeResult, seasonal_decompose

def naive_Bayes_study(
    trnX: ndarray, 
    trnY: array, 
    tstX: ndarray, 
    tstY: array, 
    metric: str = "accuracy"
) -> tuple:
    estimators: dict = {
        "GaussianNB": GaussianNB(),
        "MultinomialNB": MultinomialNB(),
        "BernoulliNB": BernoulliNB(),
    }

    xvalues: list = []
    yvalues: list = []
    best_model = None
    best_params: dict = {"name": "", "metric": metric, "params": ()}
    best_performance = 0
    for clf in estimators:
        try:
            estimators[clf].fit(trnX, trnY)
            xvalues.append(clf)
            prdY: array = estimators[clf].predict(tstX)
            eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
            if eval - best_performance > DELTA_IMPROVE:
                best_performance: float = eval
                best_params["name"] = clf
                best_params[metric] = eval
                best_model = estimators[clf]
            yvalues.append(eval)
            # print(f'NB {clf}')
        except Exception:
            print(f"Couldn't run {clf}")
            continue
    plot_bar_chart(
        xvalues,
        yvalues,
        title=f"Naive Bayes Models ({metric})",
        ylabel=metric,
        percentage=True,
    )

    return best_model, best_params

def logistic_regression_study(
    trnX: ndarray,
    trnY: array,
    tstX: ndarray,
    tstY: array,
    nr_max_iterations: int = 2500,
    lag: int = 500,
    metric: str = "accuracy",
) -> tuple[LogisticRegression | None, dict]:
    nr_iterations: list[int] = [lag] + [
        i for i in range(2 * lag, nr_max_iterations + 1, lag)
    ]

    penalty_types: list[str] = ["l1", "l2"]  # only available if optimizer='liblinear'

    best_model = None
    best_params: dict = {"name": "LR", "metric": metric, "params": ()}
    best_performance: float = 0.0

    values: dict = {}
    for type in penalty_types:
        warm_start = False
        y_tst_values: list[float] = []
        for j in range(len(nr_iterations)):
            clf = LogisticRegression(
                penalty=type,
                max_iter=lag,
                warm_start=warm_start,
                solver="liblinear",
                verbose=False,
            )
            clf.fit(trnX, trnY)
            prdY: array = clf.predict(tstX)
            eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
            y_tst_values.append(eval)
            warm_start = True
            if eval - best_performance > DELTA_IMPROVE:
                best_performance = eval
                best_params["params"] = (type, nr_iterations[j])
                best_model: LogisticRegression = clf
            # print(f'MLP lr_type={type} lr={lr} n={nr_iterations[j]}')
        values[type] = y_tst_values
    plot_multiline_chart(
        nr_iterations,
        values,
        title=f"LR models ({metric})",
        xlabel="nr iterations",
        ylabel=metric,
        percentage=True,
    )
    print(f'LR best for {best_params["params"][1]} iterations (penalty={best_params["params"][0]})')

    return best_model, best_params

def knn_study(
    trnX: ndarray, 
    trnY: array, 
    tstX: ndarray, 
    tstY: array, 
    k_max: int = 19, 
    lag: int = 2, 
    metric: str = "accuracy",
    verbose: bool = False,
) -> tuple[KNeighborsClassifier | None, dict]:
    dist: list[Literal['manhattan', 'euclidean', 'chebyshev']] = [
        'manhattan', 'euclidean', 'chebyshev'
    ]

    kvalues: list[int] = [i for i in range(1, k_max + 1, lag)]
    best_model: KNeighborsClassifier | None = None
    best_params: dict = {'name': 'KNN', 'metric': metric, 'params': ()}
    best_performance: float = 0.0

    values: dict[str, list] = {}

    # progress bookkeeping
    total_configs = len(dist) * len(kvalues)
    current_config = 0

    for d in dist:
        y_tst_values: list = []
        for k in kvalues:
            current_config += 1
            if verbose:
                print(f"[{current_config}/{total_configs}] KNN: metric={d}, k={k}")

            clf = KNeighborsClassifier(n_neighbors=k, metric=d)
            clf.fit(trnX, trnY)
            prdY: array = clf.predict(tstX)
            eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
            y_tst_values.append(eval)

            if eval - best_performance > DELTA_IMPROVE:
                best_performance = eval
                best_params['params'] = (k, d)
                best_model = clf
                if verbose:
                    print(f"    -> new best {metric}={eval:.4f}")

        values[d] = y_tst_values

    print(f'KNN best with k={best_params["params"][0]} and {best_params["params"][1]}')
    plot_multiline_chart(
        kvalues, values,
        title=f'KNN Models ({metric})',
        xlabel='k', ylabel=metric, percentage=True
    )

    return best_model, best_params

def trees_study(
    trnX: ndarray, 
    trnY: array, 
    tstX: ndarray, 
    tstY: array, 
    d_max: int=10, 
    lag:int=2, 
    metric='accuracy'
) -> tuple:
    criteria: list[Literal['entropy', 'gini']] = ['entropy', 'gini']
    depths: list[int] = [i for i in range(2, d_max+1, lag)]

    best_model: DecisionTreeClassifier | None = None
    best_params: dict = {'name': 'DT', 'metric': metric, 'params': ()}
    best_performance: float = 0.0

    values: dict = {}
    for c in criteria:
        y_tst_values: list[float] = []
        for d in depths:
            clf = DecisionTreeClassifier(max_depth=d, criterion=c, min_impurity_decrease=0)
            clf.fit(trnX, trnY)
            prdY: array = clf.predict(tstX)
            eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
            y_tst_values.append(eval)
            if eval - best_performance > DELTA_IMPROVE:
                best_performance = eval
                best_params['params'] = (c, d)
                best_model = clf
            # print(f'DT {c} and d={d}')
        values[c] = y_tst_values
    print(f'DT best with {best_params["params"][0]} and d={best_params["params"][1]}')
    plot_multiline_chart(depths, values, title=f'DT Models ({metric})', xlabel='d', ylabel=metric, percentage=True)

    return best_model, best_params

def mlp_study(
    trnX: ndarray,
    trnY: array,
    tstX: ndarray,
    tstY: array,
    nr_max_iterations: int = 2500,
    lag: int = 500,
    verbose: bool = False,
    metric: str = "accuracy",
) -> tuple[MLPClassifier | None, dict]:
    nr_iterations: list[int] = [lag] + [
        i for i in range(2 * lag, nr_max_iterations + 1, lag)
    ]

    lr_types: list[Literal["constant", "invscaling", "adaptive"]] = [
        "constant",
        "invscaling",
        "adaptive",
    ]  # only used if optimizer='sgd'
    learning_rates: list[float] = [0.5, 0.05, 0.005, 0.0005]

    best_model: MLPClassifier | None = None
    best_params: dict = {"name": "MLP", "metric": metric, "params": ()}
    best_performance: float = 0.0

    values: dict = {}
    _, axs = subplots(
        1, len(lr_types), figsize=(len(lr_types) * HEIGHT, HEIGHT), squeeze=False
    )
    for i in range(len(lr_types)):
        type: str = lr_types[i]
        values = {}
        for lr in learning_rates:
            warm_start: bool = False
            y_tst_values: list[float] = []
            for j in range(len(nr_iterations)):
                clf = MLPClassifier(
                    learning_rate=type,
                    learning_rate_init=lr,
                    max_iter=lag,
                    warm_start=warm_start,
                    activation="logistic",
                    solver="sgd",
                    verbose=verbose,
                )
                clf.fit(trnX, trnY)
                prdY: array = clf.predict(tstX)
                eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
                y_tst_values.append(eval)
                warm_start = True
                if eval - best_performance > DELTA_IMPROVE:
                    best_performance = eval
                    best_params["params"] = (type, lr, nr_iterations[j])
                    best_model = clf
                # print(f'MLP lr_type={type} lr={lr} n={nr_iterations[j]}')
            values[lr] = y_tst_values
        plot_multiline_chart(
            nr_iterations,
            values,
            ax=axs[0, i],
            title=f"MLP with {type}",
            xlabel="nr iterations",
            ylabel=metric,
            percentage=True,
        )
    print(
        f'MLP best for {best_params["params"][2]} iterations (lr_type={best_params["params"][0]} and lr={best_params["params"][1]}'
    )

    return best_model, best_params


def mlp_study_forecast(
    trnX: ndarray,
    trnY: array,
    tstX: ndarray,
    tstY: array,
    nr_max_iterations: int = 2500,
    lag: int = 500,
    verbose: bool = False,
    measure: str = "R2",
) -> tuple[MLPRegressor | None, dict]:
    nr_iterations: list[int] = [lag] + [
        i for i in range(2 * lag, nr_max_iterations + 1, lag)
    ]

    lr_types: list[Literal["constant", "invscaling", "adaptive"]] = [
        "constant",
        "invscaling",
        "adaptive",
    ]  # only used if optimizer='sgd'
    learning_rates: list[float] = [0.5, 0.05, 0.005, 0.0005]

    best_model: MLPRegressor | None = None
    best_params: dict = {"name": "MLP", "measure": measure, "params": ()}
    best_performance: float = 0.0

    values: dict = {}
    _, axs = subplots(
        1, len(lr_types), figsize=(len(lr_types) * HEIGHT, HEIGHT), squeeze=False
    )
    for i in range(len(lr_types)):
        type: str = lr_types[i]
        values = {}
        for lr in learning_rates:
            warm_start: bool = False
            y_tst_values: list[float] = []
            for j in range(len(nr_iterations)):
                clf = MLPRegressor(
                    learning_rate=type,
                    learning_rate_init=lr,
                    max_iter=lag,
                    warm_start=warm_start,
                    activation="logistic",
                    solver="sgd",
                    verbose=verbose,
                )
                clf.fit(trnX, trnY)
                prdY: array = clf.predict(tstX)
                eval: float = FORECAST_MEASURES[measure](tstY, prdY)
                y_tst_values.append(eval)
                warm_start = True
                if eval - best_performance > DELTA_IMPROVE:
                    best_performance = eval
                    best_params["params"] = (type, lr, nr_iterations[j])
                    best_model = clf
            values[lr] = y_tst_values
        plot_multiline_chart(
            nr_iterations,
            values,
            ax=axs[0, i],
            title=f"MLP with {type}",
            xlabel="nr iterations",
            ylabel=measure,
            percentage=True,
        )
    print(
        f'MLP best for {best_params["params"][2]} iterations (lr_type={best_params["params"][0]} and lr={best_params["params"][1]}'
    )

    return best_model, best_params

def mlp_study_forecast_univariate(
    train,
    test,
    measure: str = "RMSE",
    lag_values=(1, 2),
    hidden_units=(5, 10, 25, 50, 100),
    nr_max_iterations: int = 2000,
    step: int = 200,
    verbose: bool = False,
    random_state: int = 1,
):
    iterations = [step] + [i for i in range(2 * step, nr_max_iterations + 1, step)]

    minimize = measure in ("MAPE", "MAE", "MSE", "RMSE")
    best_performance = float("inf") if minimize else -float("inf")
    best_model = None
    best_params = {"name": "MLP", "metric": measure, "params": ()}

    _, axs = subplots(1, len(lag_values), figsize=(len(lag_values) * HEIGHT, HEIGHT), squeeze=False)

    for i, lag in enumerate(lag_values):
        X_trn, y_trn, X_tst, y_tst = make_supervised_from_train_test(train, test, lag=lag)
        if len(y_tst) == 0 or len(y_trn) == 0:
            continue

        values = {}
        for h in hidden_units:
            yvals = []
            clf = MLPRegressor(
                hidden_layer_sizes=(h,),
                solver="adam",
                activation="relu",
                max_iter=step,
                warm_start=True,
                random_state=random_state,
                verbose=verbose,
            )

            total = 0
            for _ in iterations:
                clf.max_iter = step
                clf.fit(X_trn, y_trn)
                total += step

                prd = clf.predict(X_tst)
                score = FORECAST_MEASURES[measure](y_tst, prd)
                yvals.append(score)

                improved = (score < best_performance - DELTA_IMPROVE) if minimize else (score > best_performance + DELTA_IMPROVE)
                if improved:
                    best_performance = score
                    best_params["params"] = (lag, h, total)
                    best_model = deepcopy(clf)

            values[h] = yvals

        plot_multiline_chart(
            iterations,
            values,
            ax=axs[0, i],
            title=f"MLP lag={lag} ({measure})",
            xlabel="nr iterations",
            ylabel=measure,
            percentage=(measure == "MAPE"),
        )

    print(f"MLP best params={best_params['params']} -> {measure}={best_performance}")
    return best_model, best_params

def make_supervised_from_train_test(train, test, lag: int):
    full = np.concatenate([np.asarray(train, dtype=float),
                           np.asarray(test, dtype=float)])
    n_train = len(train)
    n_full = len(full)

    X_trn, y_trn = [], []
    X_tst, y_tst = [], []

    for t in range(lag, n_full):
        x = full[t-lag:t]
        y = full[t]
        if t < n_train:
            X_trn.append(x); y_trn.append(y)
        else:
            X_tst.append(x); y_tst.append(y)

    return (np.asarray(X_trn), np.asarray(y_trn),
            np.asarray(X_tst), np.asarray(y_tst))

def mlp_study_tuned_for_flight(
    trnX: ndarray,
    trnY: array,
    tstX: ndarray,
    tstY: array,
    nr_max_iterations: int = 2500,
    lag: int = 500,
    verbose: bool = False,
    metric: str = "recall",
) -> tuple[MLPClassifier | None, dict]:
    nr_iterations: list[int] = [lag] + [
        i for i in range(2 * lag, nr_max_iterations + 1, lag)
    ]

    lr_types: list[Literal["constant", "invscaling", "adaptive"]] = [
        "constant",
        "invscaling",
        "adaptive",
    ]  # only used if optimizer='sgd'
    learning_rates: list[float] = [0.5, 0.05, 0.005]

    best_model: MLPClassifier | None = None
    best_params: dict = {"name": "MLP", "metric": metric, "params": ()}
    best_performance: float = 0.0

    values: dict = {}
    _, axs = subplots(
        1, len(lr_types), figsize=(len(lr_types) * HEIGHT, HEIGHT), squeeze=False
    )
    total_configs = len(lr_types) * len(learning_rates) * len(nr_iterations)
    current_config = 0
    
    for i in range(len(lr_types)):
        type: str = lr_types[i]
        values = {}
        for lr in learning_rates:
            warm_start: bool = False
            y_tst_values: list[float] = []
            for j in range(len(nr_iterations)):
                current_config += 1
                print(f"\n[{current_config}/{total_configs}] Training MLP: lr_type={type}, lr={lr}, iterations={nr_iterations[j]}")
                
                clf = MLPClassifier(
                    learning_rate=type,
                    learning_rate_init=lr,
                    max_iter=lag,
                    warm_start=warm_start,
                    activation="logistic",
                    solver="sgd",
                    verbose=verbose,
                )
                clf.fit(trnX, trnY)
                prdY: array = clf.predict(tstX)
                eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
                y_tst_values.append(eval)
                warm_start = True
                
                print(f"  -> {metric}: {eval:.4f}", end="")
                if eval - best_performance > DELTA_IMPROVE:
                    best_performance = eval
                    best_params["params"] = (type, lr, nr_iterations[j])
                    best_model = clf
                    print(f" ✓ NEW BEST!")
                else:
                    print()
                    
            values[lr] = y_tst_values
        plot_multiline_chart(
            nr_iterations,
            values,
            ax=axs[0, i],
            title=f"MLP with {type}",
            xlabel="nr iterations",
            ylabel=metric,
            percentage=True,
        )
    print(
        f'MLP best for {best_params["params"][2]} iterations (lr_type={best_params["params"][0]} and lr={best_params["params"][1]}'
    )

    return best_model, best_params

def random_forests_study(
    trnX: ndarray,
    trnY: array,
    tstX: ndarray,
    tstY: array,
    nr_max_trees: int = 2500,
    lag: int = 500,
    metric: str = "accuracy",
) -> tuple[RandomForestClassifier | None, dict]:
    n_estimators: list[int] = [100] + [i for i in range(500, nr_max_trees + 1, lag)]
    max_depths: list[int] = [2, 5, 7]
    max_features: list[float] = [0.1, 0.3, 0.5, 0.7, 0.9]

    best_model: RandomForestClassifier | None = None
    best_params: dict = {"name": "RF", "metric": metric, "params": ()}
    best_performance: float = 0.0

    values: dict = {}

    cols: int = len(max_depths)
    _, axs = subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)
    for i in range(len(max_depths)):
        d: int = max_depths[i]
        values = {}
        for f in max_features:
            y_tst_values: list[float] = []
            for n in n_estimators:
                clf = RandomForestClassifier(
                    n_estimators=n, max_depth=d, max_features=f
                )
                clf.fit(trnX, trnY)
                prdY: array = clf.predict(tstX)
                eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
                y_tst_values.append(eval)
                if eval - best_performance > DELTA_IMPROVE:
                    best_performance = eval
                    best_params["params"] = (d, f, n)
                    best_model = clf
                # print(f'RF d={d} f={f} n={n}')
            values[f] = y_tst_values
        plot_multiline_chart(
            n_estimators,
            values,
            ax=axs[0, i],
            title=f"Random Forests with max_depth={d}",
            xlabel="nr estimators",
            ylabel=metric,
            percentage=True,
        )
    print(
        f'RF best for {best_params["params"][2]} trees (d={best_params["params"][0]} and f={best_params["params"][1]})'
    )
    return best_model, best_params

def gradient_boosting_study(
    trnX: ndarray,
    trnY: array,
    tstX: ndarray,
    tstY: array,
    nr_max_trees: int = 2500,
    lag: int = 500,
    metric: str = "accuracy",
    verbose: bool = False,              # 👈 NEW
) -> tuple[GradientBoostingClassifier | None, dict]:
    n_estimators: list[int] = [100] + [i for i in range(500, nr_max_trees + 1, lag)]
    max_depths: list[int] = [2, 5, 7]
    learning_rates: list[float] = [0.1, 0.3, 0.5, 0.7, 0.9]

    best_model: GradientBoostingClassifier | None = None
    best_params: dict = {"name": "GB", "metric": metric, "params": ()}
    best_performance: float = 0.0

    values: dict = {}
    cols: int = len(max_depths)
    _, axs = subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)

    # progress bookkeeping
    total_configs = len(max_depths) * len(learning_rates) * len(n_estimators)
    current_config = 0

    for i in range(len(max_depths)):
        d: int = max_depths[i]
        values = {}
        for lr in learning_rates:
            y_tst_values: list[float] = []
            for n in n_estimators:
                current_config += 1
                if verbose:
                    print(
                        f"[{current_config}/{total_configs}] "
                        f"GB: max_depth={d}, lr={lr}, n_estimators={n}"
                    )

                clf = GradientBoostingClassifier(
                    n_estimators=n,
                    max_depth=d,
                    learning_rate=lr,
                    verbose=int(verbose),   # 👈 optional: sklearn’s own logs
                )
                clf.fit(trnX, trnY)
                prdY: array = clf.predict(tstX)
                eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
                y_tst_values.append(eval)

                if eval - best_performance > DELTA_IMPROVE:
                    best_performance = eval
                    best_params["params"] = (d, lr, n)
                    best_model = clf
                    if verbose:
                        print(
                            f"    -> new best {metric}={eval:.4f} "
                            f"(depth={d}, lr={lr}, n={n})"
                        )

            values[lr] = y_tst_values

        plot_multiline_chart(
            n_estimators,
            values,
            ax=axs[0, i],
            title=f"Gradient Boosting with max_depth={d}",
            xlabel="nr estimators",
            ylabel=metric,
            percentage=True,
        )

    print(
        f'GB best for {best_params["params"][2]} trees '
        f'(d={best_params["params"][0]} and lr={best_params["params"][1]}'
    )

    return best_model, best_params

def get_path_aux(lab_folder: str):
    if "lab3" in lab_folder:
        return "../.."
    elif "lab1" in lab_folder or "lab4" in lab_folder:
        return ".."

def save_train_test(
    df: DataFrame, 
    target: str, 
    lab_folder: str,
    file_tag: str,
    approach: str,
    test_size: float = 0.3,
    random_state: int = 42
):
    data = df.copy()
    labels: list = list(data[target].unique())
    labels.sort()

    positive: int = 1
    negative: int = 0
    
    values: dict[str, list[int]] = {
        "Original": [
            len(data[data[target] == negative]),
            len(data[data[target] == positive]),
        ]
    }

    y = data.pop(target).values
    X = data.values
    
    # Split the data
    trnX, tstX, trnY, tstY = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    train: DataFrame = concat(
        [DataFrame(trnX, columns=data.columns), DataFrame(trnY, columns=[target])], axis=1
    )
    train.to_csv(f'{get_path_aux(lab_folder)}/data/prepared/{file_tag}_{approach}_train.csv', index=False)

    test: DataFrame = concat(
        [DataFrame(tstX, columns=data.columns), DataFrame(tstY, columns=[target])], axis=1
    )
    test.to_csv(f'{get_path_aux(lab_folder)}/data/prepared/{file_tag}_{approach}_test.csv', index=False)

    values["Train"] = [
        len(train[train[target] == negative]),
        len(train[train[target] == positive]),
    ]
    values["Test"] = [
        len(test[test[target] == negative]),
        len(test[test[target] == positive]),
    ]

    figure(figsize=(6, 4))
    plot_multibar_chart(labels, values, title="Data distribution per dataset")
    show()
    return train, test, labels


# Made for flight dataset where shuffling is not desired
def separate_train_test_no_shuffle(
    data: DataFrame, 
    target: str, 
    test_size: float = 0.3,
):
    df = data.copy()
    
    y = df.pop(target)
    X = df
    
    # Split the data
    split_index = int((1 - test_size) * len(df))
    trnX, tstX = X.iloc[:split_index].copy(), X.iloc[split_index:].copy()
    trnY, tstY = y.iloc[:split_index].copy(), y.iloc[split_index:].copy()
    
    return trnX, tstX, trnY, tstY


# Made for flight dataset where shuffling is not desired
# should use the one above, delete it
def evaluate_approach_no_shuffle(
    data: DataFrame, 
    target: str = "class", 
    metric: str = "recall",
    test_size: float = 0.3,
) -> dict[str, list]:
    df = data.copy()
    
    y = df.pop(target).values
    X = df.values
    
    # Split the data
    split_index = int((1 - test_size) * len(df))
    trnX, tstX = X[:split_index], X[split_index:]
    trnY, tstY = y[:split_index], y[split_index:]
    
    eval: dict[str, list] = {}

    eval_NB: dict[str, float] = run_NB(trnX, trnY, tstX, tstY, metric=metric)
    eval_KNN: dict[str, float] = run_KNN(trnX, trnY, tstX, tstY, metric=metric)
    
    if eval_NB != {} and eval_KNN != {}:
        for met in CLASS_EVAL_METRICS:
            eval[met] = [eval_NB[met], eval_KNN[met]]
        eval["confusion_matrix"] = [eval_NB["confusion_matrix"], eval_KNN["confusion_matrix"]]
    
    return eval

def evaluate_and_plot(
    train_df: DataFrame, 
    test_df: DataFrame,
    lab_folder: str,
    file_tag: str,
    approach: str,
    target_name: str = "class",
    metric: str = "recall"
) -> None:
    train = train_df.copy()
    test = test_df.copy()
    labels = list(train[target_name].unique())
    figure()
    eval: dict[str, list] = evaluate_approach(train=train, test=test, target=target_name, metric=metric)
    plot_multibar_chart(
        ["NB", "KNN"], {k: v for k, v in eval.items() if k != "confusion_matrix"}, title=f"{file_tag}-{approach} evaluation", percentage=True
    )
    savefig(f"../../charts/{lab_folder}/{file_tag}_{approach}_nb_vs_knn_performance.png", bbox_inches='tight')
    show()

    fig, axs = subplots(1, 2, figsize=(2 * HEIGHT, HEIGHT))
    plot_confusion_matrix(eval["confusion_matrix"][0], labels, ax=axs[0])
    axs[0].set_title(f"{file_tag}-{approach} NB Confusion Matrix")
    plot_confusion_matrix(eval["confusion_matrix"][1], labels, ax=axs[1])
    axs[1].set_title(f"{file_tag}-{approach} KNN Confusion Matrix")
    savefig(f"../../charts/{lab_folder}/{file_tag}_{approach}_nb_vs_knn_confusion_matrix.png", bbox_inches='tight')
    show()

def predict_and_eval(features_train: ndarray,
    target_train: array,
    features_test: ndarray,
    target_test: array,
    best_model: DataFrame,
    params: DataFrame,
    lab_folder: str,
    file_tag: str,
    approach: str
) -> None:
    prd_trn = best_model.predict(features_train)
    prd_tst = best_model.predict(features_test)
    nb_labels = sorted(np.unique(target_train))

    figure()
    plot_evaluation_results(
        params,
        array(target_train),
        array(prd_trn),
        array(target_test),
        array(prd_tst),
        nb_labels
    )
    savefig(f'{get_path_aux(lab_folder)}/charts/{lab_folder}/{file_tag}_{approach}_{params["name"]}_best_{params["metric"]}_eval.png', bbox_inches='tight')
    show()

### Classification strategies

def run_all_nb(features_train: ndarray,
    target_train: array,
    features_test: ndarray,
    target_test: array,
    lab_folder: str,
    file_tag: str,
    approach: str,
    eval_metric: str,
):  
    figure()
    best_model, params = naive_Bayes_study(
        features_train,
        target_train,
        features_test,
        target_test,
        metric=eval_metric
    )
    savefig(f'{get_path_aux(lab_folder)}/charts/{lab_folder}/{file_tag}_{approach}_{params["name"]}_{params["metric"]}_study.png', bbox_inches='tight')
    show()

    predict_and_eval(features_train, target_train, features_test, target_test, 
        best_model, params, lab_folder, file_tag, approach)                    
    return best_model, params                    

def best_model_knn(
    features_train: ndarray,
    target_train: array,
    features_test: ndarray,
    target_test: array,
    lab_folder: str,
    file_tag: str,
    approach: str,
    k_max: int = 25,
    lag: int = 2,
    eval_metric: str = "accuracy",
    verbose: bool = False,
) :
    figure()
    knn_best_model, knn_params = knn_study(
        features_train,
        target_train,
        features_test,
        target_test,
        k_max,
        lag,
        metric=eval_metric,
        verbose=verbose,
    )
    savefig(f'{get_path_aux(lab_folder)}/charts/{lab_folder}/{file_tag}_{approach}_{knn_params["name"]}_{knn_params["metric"]}_study.png', bbox_inches='tight')
    show()     
    return knn_best_model, knn_params  

def knn_overfitting(
    features_train: ndarray,
    target_train: array,
    features_test: ndarray,
    target_test: array,
    params: dict,
    lab_folder: str,
    file_tag: str,
    approach: str,
    k_max: int,
    lag: int,
    eval_metric: str
):
    distance = params["params"][1]   # best distance from KNN study, e.g. 'euclidean'
    kvalues = [i for i in range(1, k_max + 1, lag)]

    y_tst_values = []
    y_trn_values = []
    
    for k in kvalues:
        clf = KNeighborsClassifier(n_neighbors=k, metric=distance)
        clf.fit(features_train, target_train)
        prd_tst_Y = clf.predict(features_test)
        prd_trn_Y = clf.predict(features_train)

        y_tst_values.append(CLASS_EVAL_METRICS[eval_metric](target_test, prd_tst_Y))
        y_trn_values.append(CLASS_EVAL_METRICS[eval_metric](target_train, prd_trn_Y))
                        
    figure()
    plot_multiline_chart(
        kvalues,
        {"Train": y_trn_values, "Test": y_tst_values},
        title=f"KNN overfitting study for {distance}",
        xlabel="K",
        ylabel=eval_metric,
        percentage=True,
    )
    savefig(f'{get_path_aux(lab_folder)}/charts/{lab_folder}/{file_tag}_{approach}_{params["name"]}_{params["metric"]}_overfitting.png', bbox_inches='tight')
    show()

def run_all_knn(features_train: ndarray,
    target_train: array,
    features_test: ndarray,
    target_test: array,
    lab_folder: str,
    file_tag: str,
    approach: str,
    k_max: int,
    lag: int,
    eval_metric: str,
    verbose: bool = False,
):  
    best_model, params = best_model_knn(
        features_train, target_train, features_test, target_test, 
        lab_folder, file_tag, approach,
        k_max=k_max,
        lag=lag,
        eval_metric = eval_metric,
        verbose=verbose,
    )
    if verbose:
        print("\nRunning KNN overfitting study...")
    knn_overfitting(features_train, target_train, features_test, target_test, 
        params, lab_folder, file_tag, approach,
        k_max=k_max,
        lag=lag,
        eval_metric = eval_metric
    ) 
    predict_and_eval(features_train, target_train, features_test, target_test, 
        best_model, params, lab_folder, file_tag, approach)                    
    return best_model, params                    

def best_model_lr(
    features_train: ndarray,
    target_train: array,
    features_test: ndarray,
    target_test: array,
    lab_folder: str,
    file_tag: str,
    approach: str,
    nr_max_iterations: int,
    lag: int,
    eval_metric: str
) :
    figure()
    lr_best_model, lr_params = logistic_regression_study(
        features_train,
        target_train,
        features_test,
        target_test,
        nr_max_iterations=nr_max_iterations,
        lag=lag,
        metric=eval_metric,
    )
    savefig(f'{get_path_aux(lab_folder)}/charts/{lab_folder}/{file_tag}_{approach}_{lr_params["name"]}_{lr_params["metric"]}_study.png', bbox_inches='tight')
    show()     
    return lr_best_model, lr_params  

def lr_overfitting(
    features_train: ndarray,
    target_train: array,
    features_test: ndarray,
    target_test: array,
    params: dict,
    lab_folder: str,
    file_tag: str,
    approach: str,
    nr_max_iterations: int,
    lag: int,
    eval_metric: str
):  
    type: str = params["params"][0]
    nr_iterations: list[int] = [i for i in range(lag, nr_max_iterations+1, lag)]

    y_tst_values: list[float] = []
    y_trn_values: list[float] = []
    
    warm_start = False
    for n in nr_iterations:
        clf = LogisticRegression(
            warm_start=warm_start,
            penalty=type,
            max_iter=n,
            solver="liblinear",
            verbose=False,
        )
        clf.fit(features_train, target_train)
        prd_tst_Y: array = clf.predict(features_test)
        prd_trn_Y: array = clf.predict(features_train)
        y_tst_values.append(CLASS_EVAL_METRICS[eval_metric](target_test, prd_tst_Y))
        y_trn_values.append(CLASS_EVAL_METRICS[eval_metric](target_train, prd_trn_Y))
        warm_start = True

    figure()
    plot_multiline_chart(
        nr_iterations,
        {"Train": y_trn_values, "Test": y_tst_values},
        title=f"LR overfitting study for penalty={type}",
        xlabel="nr_iterations",
        ylabel=str(eval_metric),
        percentage=True,
    )
    savefig(f'{get_path_aux(lab_folder)}/charts/{lab_folder}/{file_tag}_{approach}_{params["name"]}_{params["metric"]}_overfitting.png', bbox_inches='tight')
    show()

def run_all_lr(
    features_train: ndarray,
    target_train: array,
    features_test: ndarray,
    target_test: array,
    lab_folder: str,
    file_tag: str,
    approach: str,
    nr_max_iterations: int,
    lag: int,
    eval_metric: str
):  
    best_model, params = best_model_lr(
        features_train, target_train, features_test, target_test, 
        lab_folder, file_tag, approach,
        nr_max_iterations=nr_max_iterations,
        lag=lag,
        eval_metric = eval_metric
    )
    lr_overfitting(features_train, target_train, features_test, target_test, 
        params, lab_folder, file_tag, approach,
        nr_max_iterations=nr_max_iterations,
        lag=lag,
        eval_metric = eval_metric
    ) 
    predict_and_eval(features_train, target_train, features_test, target_test, 
        best_model, params, lab_folder, file_tag, approach)                    
    return best_model, params                    

def dt_overfitting(
    features_train: ndarray,
    target_train: array,
    features_test: ndarray,
    target_test: array,
    params: dict,
    lab_folder: str,
    file_tag: str,
    approach: str,
    d_max: int,
    lag: int,
    eval_metric: str
):
    crit = params["params"][0]   # 'entropy' or 'gini'
    depths = [i for i in range(2, d_max + 1, lag)]

    y_tst_values = []
    y_trn_values = []

    for d in depths:
        clf = DecisionTreeClassifier(
            max_depth=d,
            criterion=crit,
            min_impurity_decrease=0,
            random_state=42,
        )
        clf.fit(features_train, target_train)
        prd_tst_Y = clf.predict(features_test)
        prd_trn_Y = clf.predict(features_train)

        y_tst_values.append(CLASS_EVAL_METRICS[eval_metric](target_test, prd_tst_Y))
        y_trn_values.append(CLASS_EVAL_METRICS[eval_metric](target_train, prd_trn_Y))

    figure()
    plot_multiline_chart(
        depths,
        {"Train": y_trn_values, "Test": y_tst_values},
        title=f"DT overfitting study for {crit}",
        xlabel="max_depth",
        ylabel=eval_metric,
        percentage=True,
    )
    savefig(f'{get_path_aux(lab_folder)}/charts/{lab_folder}/{file_tag}_{approach}_{params["name"]}_{params["metric"]}_overfitting.png', bbox_inches='tight')
    show()

def run_all_dt(
    features_train: ndarray,
    target_train: array,
    features_test: ndarray,
    target_test: array,
    lab_folder: str,
    file_tag: str,
    approach: str,
    d_max: int,
    lag: int,
    eval_metric: str
):  
    figure()
    best_model, params = trees_study(
        features_train,
        target_train,
        features_test,
        target_test,
        d_max=d_max,
        lag=lag,
        metric=eval_metric,
    )
    savefig(f'{get_path_aux(lab_folder)}/charts/{lab_folder}/{file_tag}_{approach}_{params["name"]}_{params["metric"]}_study.png', bbox_inches='tight')
    show()
    dt_overfitting(features_train, target_train, features_test, target_test, 
        params, lab_folder, file_tag, approach,
        d_max=d_max,
        lag=lag,
        eval_metric = eval_metric
    ) 
    predict_and_eval(features_train, target_train, features_test, target_test, 
        best_model, params, lab_folder, file_tag, approach)
    return best_model, params                    

def show_tree_and_importances_dt(
    features: DataFrame,
    target: DataFrame, 
    dt_best_model: DataFrame, 
    dt_params: dict,
    lab_folder: str,
    file_tag: str,
    approach: str,
    max_depth2show: int
):
    dt_feature_names = list(map(str, list(features.columns)))

    dt_class_names = list(map(str, sorted(target.unique())))

    figure(figsize=(18, 10))
    plot_tree(
        dt_best_model,
        max_depth=max_depth2show,
        feature_names=dt_feature_names,
        class_names=dt_class_names,
        filled=True,
        rounded=True,
        impurity=False,
        precision=2,
    )
    savefig(f'{get_path_aux(lab_folder)}/charts/{lab_folder}/{file_tag}_{approach}_{dt_params["name"]}_{dt_params["metric"]}_tree_depth{max_depth2show}.png', bbox_inches='tight')
    show()

    importances = dt_best_model.feature_importances_

    indices = argsort(importances)[::-1]
    dt_vars = list(features.columns)
    elems = []
    imp_values = []

    # print ranked list like professor
    for f in range(len(dt_vars)):
        feature_name = dt_vars[indices[f]]
        feature_imp = importances[indices[f]]

        elems.append(feature_name)
        imp_values.append(feature_imp)

        print(f"{f+1}. {feature_name} ({feature_imp})")

    figure()
    plot_horizontal_bar_chart(
        elems,
        imp_values,
        title="Decision Tree variables importance",
        xlabel="importance",
        ylabel="variables",
        percentage=True,
    )
    savefig(f'{get_path_aux(lab_folder)}/charts/{lab_folder}/{file_tag}_{approach}_{dt_params["name"]}_{dt_params["metric"]}_importance_ranking.png', bbox_inches='tight')
    show()

def mlp_overfitting(
    features_train: ndarray,
    target_train: array,
    features_test: ndarray,
    target_test: array,
    params: dict,
    lab_folder: str,
    file_tag: str,
    approach: str,
    nr_max_iterations: int,
    lag: int,
    eval_metric: str
):
    lr_type = params["params"][0]
    lr = params["params"][1]

    nr_iterations = [i for i in range(lag, nr_max_iterations + 1, lag)]

    y_tst_values = []
    y_trn_values = []
    
    for n in nr_iterations:
        clf = MLPClassifier(
            learning_rate=lr_type,
            learning_rate_init=lr,
            max_iter=n,
            activation="logistic",
            solver="adam",
            verbose=False,
            random_state=42,
        )
        clf.fit(features_train, target_train)
        prd_tst_Y = clf.predict(features_test)
        prd_trn_Y = clf.predict(features_train)

        y_tst_values.append(CLASS_EVAL_METRICS[eval_metric](target_test, prd_tst_Y))
        y_trn_values.append(CLASS_EVAL_METRICS[eval_metric](target_train, prd_trn_Y))

    figure()
    plot_multiline_chart(
        nr_iterations,
        {"Train": y_trn_values, "Test": y_tst_values},
        title=f"MLP overfitting study for lr_type={lr_type} and lr={lr}",
        xlabel="nr_iterations",
        ylabel=eval_metric,
        percentage=True,
    )
    savefig(f'{get_path_aux(lab_folder)}/charts/{lab_folder}/{file_tag}_{approach}_{params["name"]}_{params["metric"]}_overfitting.png', bbox_inches='tight')
    show()

def mlp_overfitting_for_flight(
    features_train: ndarray,
    target_train: array,
    features_test: ndarray,
    target_test: array,
    params: dict,
    lab_folder: str,
    file_tag: str,
    approach: str,
    nr_max_iterations: int,
    lag: int,
    eval_metric: str
):
    lr_type = params["params"][0]
    lr = params["params"][1]

    nr_iterations = [i for i in range(lag, nr_max_iterations + 1, lag)]

    y_tst_values = []
    y_trn_values = []
    
    for n in nr_iterations:
        clf = MLPClassifier(
            learning_rate=lr_type,
            learning_rate_init=lr,
            max_iter=n,
            activation="logistic",
            solver="adam",
            verbose=True,
            random_state=42,
        )
        clf.fit(features_train, target_train)
        prd_tst_Y = clf.predict(features_test)
        prd_trn_Y = clf.predict(features_train)

        y_tst_values.append(CLASS_EVAL_METRICS[eval_metric](target_test, prd_tst_Y))
        y_trn_values.append(CLASS_EVAL_METRICS[eval_metric](target_train, prd_trn_Y))

    figure()
    plot_multiline_chart(
        nr_iterations,
        {"Train": y_trn_values, "Test": y_tst_values},
        title=f"MLP overfitting study for lr_type={lr_type} and lr={lr}",
        xlabel="nr_iterations",
        ylabel=eval_metric,
        percentage=True,
    )
    savefig(f'{get_path_aux(lab_folder)}/charts/{lab_folder}/{file_tag}_{approach}_{params["name"]}_{params["metric"]}_overfitting.png', bbox_inches='tight')
    show()

def run_all_mlp(features_train: ndarray,
    target_train: array,
    features_test: ndarray,
    target_test: array,
    lab_folder: str,
    file_tag: str,
    approach: str,
    nr_max_iterations: int,
    lag: int,
    eval_metric: str,
    flight: bool = False,
    verbose: bool = False
):  
    figure()
    if flight:
        best_model, params = mlp_study_tuned_for_flight(
            features_train,
            target_train,
            features_test,
            target_test,
            nr_max_iterations=nr_max_iterations,
            lag=lag,
            metric=eval_metric,
            verbose=True,
        )
    else:
        best_model, params = mlp_study(
            features_train,
            target_train,
            features_test,
            target_test,
            nr_max_iterations=nr_max_iterations,
            lag=lag,
            metric=eval_metric,
            verbose=verbose,
        )
    savefig(f'{get_path_aux(lab_folder)}/charts/{lab_folder}/{file_tag}_{approach}_{params["name"]}_{params["metric"]}_study.png', bbox_inches='tight')
    show()
    
    if flight:        
        mlp_overfitting(features_train, target_train, features_test, target_test, 
            params, lab_folder, file_tag, approach,
            nr_max_iterations=nr_max_iterations,
            lag=lag,
            eval_metric = eval_metric
        )
    else:
       mlp_overfitting_for_flight(features_train, target_train, features_test, target_test, 
            params, lab_folder, file_tag, approach,
            nr_max_iterations=nr_max_iterations,
            lag=lag,
            eval_metric = eval_metric
        ) 
    
    figure()
    plot_line_chart(
        arange(len(best_model.loss_curve_)),
        best_model.loss_curve_,
        title="Loss curve for MLP best model training",
        xlabel="iterations",
        ylabel="loss",
        percentage=False,
    )
    savefig(f'{get_path_aux(lab_folder)}/charts/{lab_folder}/{file_tag}_{approach}_{params["name"]}_{params["metric"]}_loss_curve.png', bbox_inches='tight')
    show()

    predict_and_eval(features_train, target_train, features_test, target_test, 
        best_model, params, lab_folder, file_tag, approach)
    return best_model, params                    

def rf_overfitting(
    features_train: ndarray,
    target_train: array,
    features_test: ndarray,
    target_test: array,
    params: dict,
    lab_folder: str,
    file_tag: str,
    approach: str,
    nr_max_trees: int,
    lag: int,
    eval_metric: str
):
    d_max: int = params["params"][0]
    feat: float = params["params"][1]
    nr_estimators: list[int] = [i for i in range(2, nr_max_trees+1, lag)]

    y_tst_values: list[float] = []
    y_trn_values: list[float] = []
    
    for n in nr_estimators:
        clf = RandomForestClassifier(n_estimators=n, max_depth=d_max, max_features=feat)
        clf.fit(features_train, target_train)
        prd_tst_Y: array = clf.predict(features_test)
        prd_trn_Y: array = clf.predict(features_train)
        y_tst_values.append(CLASS_EVAL_METRICS[eval_metric](target_test, prd_tst_Y))
        y_trn_values.append(CLASS_EVAL_METRICS[eval_metric](target_train, prd_trn_Y))

    figure()
    plot_multiline_chart(
        nr_estimators,
        {"Train": y_trn_values, "Test": y_tst_values},
        title=f"RF overfitting study for d={d_max} and f={feat}",
        xlabel="nr_estimators",
        ylabel=str(eval_metric),
        percentage=True,
    )
    savefig(f'{get_path_aux(lab_folder)}/charts/{lab_folder}/{file_tag}_{approach}_{params["name"]}_{params["metric"]}_overfitting.png', bbox_inches='tight')
    show()

def show_importances_rf(
    features: DataFrame,
    best_model: DataFrame, 
    params: dict,
    lab_folder: str,
    file_tag: str,
    approach: str,
):
    stdevs: list[float] = list(
        std([tree.feature_importances_ for tree in best_model.estimators_], axis=0)
    )
    importances = best_model.feature_importances_
    rf_vars = list(features.columns)
    indices: list[int] = argsort(importances)[::-1]
    elems: list[str] = []
    imp_values: list[float] = []
    for f in range(len(rf_vars)):
        elems += [rf_vars[indices[f]]]
        imp_values.append(importances[indices[f]])
        print(f"{f+1}. {elems[f]} ({importances[indices[f]]})")

    figure()
    plot_horizontal_bar_chart(
        elems,
        imp_values,
        error=stdevs,
        title="RF variables importance",
        xlabel="importance",
        ylabel="variables",
        percentage=True,
    )
    savefig(f'{get_path_aux(lab_folder)}/charts/{lab_folder}/{file_tag}_{approach}_{params["name"]}_{params["metric"]}_vars_ranking.png', bbox_inches='tight')
    show()

def run_all_rf(features_train: ndarray,
    target_train: array,
    features_test: ndarray,
    target_test: array,
    lab_folder: str,
    file_tag: str,
    approach: str,
    nr_max_trees: int,
    lag: int,
    eval_metric: str
):  
    figure()
    best_model, params = random_forests_study(
        features_train,
        target_train,
        features_test,
        target_test,
        nr_max_trees=nr_max_trees,
        lag=lag,
        metric=eval_metric,
    )
    savefig(f'{get_path_aux(lab_folder)}/charts/{lab_folder}/{file_tag}_{approach}_{params["name"]}_{params["metric"]}_study.png', bbox_inches='tight')
    show()

    rf_overfitting(features_train, target_train, features_test, target_test, 
        params, lab_folder, file_tag, approach,
        nr_max_trees=nr_max_trees,
        lag=lag,
        eval_metric = eval_metric
    )

    predict_and_eval(features_train, target_train, features_test, target_test, 
        best_model, params, lab_folder, file_tag, approach)
    return best_model, params  

def gb_overfitting(
    features_train: ndarray,
    target_train: array,
    features_test: ndarray,
    target_test: array,
    params: dict,
    lab_folder: str,
    file_tag: str,
    approach: str,
    nr_max_trees: int,
    lag: int,
    eval_metric: str
):
    d_max: int = params["params"][0]
    lr: float = params["params"][1]
    nr_estimators: list[int] = [i for i in range(2, nr_max_trees+1, lag)]

    y_tst_values: list[float] = []
    y_trn_values: list[float] = []
    
    for n in nr_estimators:
        clf = GradientBoostingClassifier(n_estimators=n, max_depth=d_max, learning_rate=lr)
        clf.fit(features_train, target_train)
        prd_tst_Y: array = clf.predict(features_test)
        prd_trn_Y: array = clf.predict(features_train)
        y_tst_values.append(CLASS_EVAL_METRICS[eval_metric](target_test, prd_tst_Y))
        y_trn_values.append(CLASS_EVAL_METRICS[eval_metric](target_train, prd_trn_Y))

    figure()
    plot_multiline_chart(
        nr_estimators,
        {"Train": y_trn_values, "Test": y_tst_values},
        title=f"GB overfitting study for d={d_max} and lr={lr}",
        xlabel="nr_estimators",
        ylabel=str(eval_metric),
        percentage=True,
    )
    savefig(f'{get_path_aux(lab_folder)}/charts/{lab_folder}/{file_tag}_{approach}_{params["name"]}_{params["metric"]}_overfitting.png', bbox_inches='tight')
    show()

def show_importances_gb(
    features: DataFrame,
    best_model: DataFrame, 
    params: dict,
    lab_folder: str,
    file_tag: str,
    approach: str,
):
    trees_importances: list[float] = []
    for lst_trees in best_model.estimators_:
        for tree in lst_trees:
            trees_importances.append(tree.feature_importances_)

    stdevs: list[float] = list(std(trees_importances, axis=0))
    importances = best_model.feature_importances_
    gb_vars = list(features.columns)
    indices: list[int] = argsort(importances)[::-1]
    elems: list[str] = []
    imp_values: list[float] = []
    for f in range(len(gb_vars)):
        elems += [gb_vars[indices[f]]]
        imp_values.append(importances[indices[f]])
        print(f"{f+1}. {elems[f]} ({importances[indices[f]]})")

    figure()
    plot_horizontal_bar_chart(
        elems,
        imp_values,
        error=stdevs,
        title="GB variables importance",
        xlabel="importance",
        ylabel="variables",
        percentage=True,
    )
    savefig(f'{get_path_aux(lab_folder)}/charts/{lab_folder}/{file_tag}_{approach}_{params["name"]}_{params["metric"]}_vars_ranking.png', bbox_inches='tight')
    show()

def run_all_gb(
    features_train: ndarray,
    target_train: array,
    features_test: ndarray,
    target_test: array,
    lab_folder: str,
    file_tag: str,
    approach: str,
    nr_max_trees: int,
    lag: int,
    eval_metric: str,
    verbose: bool = False,
):  
    figure()
    best_model, params = gradient_boosting_study(
        features_train,
        target_train,
        features_test,
        target_test,
        nr_max_trees=nr_max_trees,
        lag=lag,
        metric=eval_metric,
        verbose=verbose,   
    )
    savefig(
        f'{get_path_aux(lab_folder)}/charts/{lab_folder}/{file_tag}_{approach}_{params["name"]}_{params["metric"]}_study.png',
        bbox_inches='tight'
    )
    show()
    
    gb_overfitting(
        features_train, target_train,
        features_test, target_test, 
        params, lab_folder, file_tag, approach,
        nr_max_trees=nr_max_trees,
        lag=lag,
        eval_metric=eval_metric,
    )

    predict_and_eval(
        features_train, target_train, features_test, target_test, 
        best_model, params, lab_folder, file_tag, approach
    )
    return best_model, params  

### Time series

def plot_ts_multivariate_chart(data: DataFrame, title: str) -> list[Axes]:
    fig: Figure
    axs: list[Axes]
    fig, axs = subplots(data.shape[1], 1, figsize=(3 * HEIGHT, HEIGHT / 2 * data.shape[1]))
    fig.suptitle(title)

    for i in range(data.shape[1]):
        col: str = data.columns[i]
        plot_line_chart(
            data[col].index.to_list(),
            data[col].to_list(),
            ax=axs[i],
            xlabel=data.index.name,
            ylabel=col,
        )
    return axs

def get_lagged_series(series: Series, max_lag: int, delta: int = 1):
    lagged_series: dict = {"original": series, "lag 1": series.shift(1)}
    for i in range(delta, max_lag + 1, delta):
        lagged_series[f"lag {i}"] = series.shift(i)
    return lagged_series

def autocorrelation_study(series: Series, max_lag: int, delta: int = 1):
    k: int = int(max_lag / delta)
    fig = figure(figsize=(4 * HEIGHT, 2 * HEIGHT), constrained_layout=True)
    gs = GridSpec(2, k, figure=fig)

    series_values: list = series.tolist()
    for i in range(1, k + 1):
        ax = fig.add_subplot(gs[0, i - 1])
        lag = i * delta
        ax.scatter(series.shift(lag).tolist(), series_values)
        ax.set_xlabel(f"lag {lag}")
        ax.set_ylabel("original")
    ax = fig.add_subplot(gs[1, :])
    ax.acorr(series, maxlags=max_lag)
    ax.set_title("Autocorrelation")
    ax.set_xlabel("Lags")
    return

def plot_components(
    series: Series,
    title: str = "",
    x_label: str = "time",
    y_label: str = "",
) -> list[Axes]:
    decomposition: DecomposeResult = seasonal_decompose(series, model="add")
    components: dict = {
        "observed": series,
        "trend": decomposition.trend,
        "seasonal": decomposition.seasonal,
        "residual": decomposition.resid,
    }
    rows: int = len(components)
    fig: Figure
    axs: list[Axes]
    fig, axs = subplots(rows, 1, figsize=(3 * HEIGHT, rows * HEIGHT))
    fig.suptitle(f"{title}")
    i: int = 0
    for key in components:
        set_chart_labels(axs[i], title=key, xlabel=x_label, ylabel=y_label)
        axs[i].plot(components[key])
        i += 1
    return fig, axs

def eval_stationarity(series: Series) -> bool:
    result = adfuller(series)
    print(f"ADF Statistic: {result[0]:.3f}")
    print(f"p-value: {result[1]:.3f}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"\t{key}: {value:.3f}")
    return result[1] <= 0.05

def scale_all_dataframe(data: DataFrame) -> DataFrame:
    vars: list[str] = data.columns.to_list()
    transf: StandardScaler = StandardScaler().fit(data)
    df = DataFrame(transf.transform(data), index=data.index)
    df.columns = vars
    return df

def series_train_test_split(data: Series, trn_pct: float = 0.90) -> tuple[Series, Series]:
    trn_size: int = int(len(data) * trn_pct)
    df_cp: Series = data.copy()
    train: Series = df_cp.iloc[:trn_size, :]
    test: Series = df_cp.iloc[trn_size:]
    return train, test

def plot_forecasting_series(
    trn: Series,
    tst: Series,
    prd_tst: Series,
    title: str = "",
    xlabel: str = "time",
    ylabel: str = "",
) -> list[Axes]:
    fig, ax = subplots(1, 1, figsize=(4 * HEIGHT, HEIGHT), squeeze=True)
    fig.suptitle(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.plot(trn.index, trn.values, label="train", color=PAST_COLOR)
    ax.plot(tst.index, tst.values, label="test", color=FUTURE_COLOR)
    ax.plot(prd_tst.index, prd_tst.values, "--", label="test prediction", color=PRED_FUTURE_COLOR)
    ax.legend(prop={"size": 5})

    return ax

def plot_forecasting_eval(trn: Series, tst: Series, prd_trn: Series, prd_tst: Series, title: str = "") -> list[Axes]:
    ev1: dict = {
        "RMSE": [sqrt(FORECAST_MEASURES["MSE"](trn, prd_trn)), sqrt(FORECAST_MEASURES["MSE"](tst, prd_tst))],
        "MAE": [FORECAST_MEASURES["MAE"](trn, prd_trn), FORECAST_MEASURES["MAE"](tst, prd_tst)],
    }
    ev2: dict = {
        "MAPE": [FORECAST_MEASURES["MAPE"](trn, prd_trn), FORECAST_MEASURES["MAPE"](tst, prd_tst)],
        "R2": [FORECAST_MEASURES["R2"](trn, prd_trn), FORECAST_MEASURES["R2"](tst, prd_tst)],
    }

    # print(eval1, eval2)
    fig, axs = subplots(1, 2, figsize=(1.5 * HEIGHT, 0.75 * HEIGHT), squeeze=True)
    fig.suptitle(title)
    plot_multibar_chart(["train", "test"], ev1, ax=axs[0], title="Scale-dependent error", percentage=False)
    plot_multibar_chart(["train", "test"], ev2, ax=axs[1], title="Percentage error", percentage=True)
    return axs


