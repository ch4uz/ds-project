from numpy import array, ndarray
from matplotlib.pyplot import subplots
from matplotlib.pyplot import figure, savefig, show, subplots
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from typing import Literal

from dslabs_functions import \
    CLASS_EVAL_METRICS, DELTA_IMPROVE, plot_bar_chart, plot_multiline_chart, \
    HEIGHT, run_NB, run_KNN, plot_multibar_chart, \
    plot_confusion_matrix

def naive_Bayes_study(
    trnX: ndarray,
    trnY: array,
    tstX: ndarray,
    tstY: array,
    metric: str = "accuracy"
):
    estimators = {
        "GaussianNB": GaussianNB(),
        "MultinomialNB": MultinomialNB(),
        "BernoulliNB": BernoulliNB(),
    }

    xvalues = []
    yvalues = []
    best_model = None
    best_params = {"name": "", "metric": metric, "params": ()}
    best_performance = 0

    for clf_name in estimators:
        try :
            estimators[clf_name].fit(trnX, trnY)
            xvalues.append(clf_name)
            prdY = estimators[clf_name].predict(tstX)
            val = CLASS_EVAL_METRICS[metric](tstY, prdY)
            if val - best_performance > DELTA_IMPROVE:
                best_performance = val
                best_params["name"] = clf_name
                best_params[metric] = val
                best_model = estimators[clf_name]
            yvalues.append(val)
        except Exception:
            print(f"Couldn't run {clf_name}")
            continue

    print(xvalues)
    print(yvalues)
    # get Axes from DSLabs helper
    ax = plot_bar_chart(
        xvalues,
        yvalues,
        title=f"Naive Bayes Models ({metric})",
        ylabel=metric,
        percentage=True,
    )

    # remove the default labels that plot_bar_chart added
    for t in ax.texts:
        t.set_visible(False)

    # add our own labels with more precision (change .6f if you want)
    for bar in ax.patches:
        height = bar.get_height()
        ax.annotate(
            f"{height:.4f}",
            (bar.get_x() + bar.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=7,
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
):
    nr_iterations = list(range(lag, nr_max_iterations + 1, lag))
    penalty_types = ["l1", "l2"]  # only valid with solver='liblinear'

    best_model = None
    best_params = {"name": "LR", "metric": metric, "params": ()}
    best_performance = 0.0

    values = {}
    for penalty in penalty_types:
        y_tst_values = []
        for n_iter in nr_iterations:
            clf = LogisticRegression(
                penalty=penalty,
                max_iter=n_iter,
                solver="liblinear",
                verbose=False,
            )
            clf.fit(trnX, trnY)
            prdY = clf.predict(tstX)
            val = CLASS_EVAL_METRICS[metric](tstY, prdY)
            y_tst_values.append(val)

            if val - best_performance > DELTA_IMPROVE:
                best_performance = val
                best_params["params"] = (penalty, n_iter)
                best_model = clf

        values[penalty] = y_tst_values

    plot_multiline_chart(
        nr_iterations,
        values,
        title=f"LR models ({metric})",
        xlabel="nr iterations",
        ylabel=metric,
        percentage=True,
    )

    print(
        f'LR best for {best_params["params"][1]} iterations '
        f'(penalty={best_params["params"][0]}) with {metric}={best_performance:.6f}'
    )

    return best_model, best_params

def knn_study(
    trnX: ndarray,
    trnY: array,
    tstX: ndarray,
    tstY: array,
    k_max: int = 19,
    lag: int = 2,
    metric: str = "accuracy",
):
    dist: list[Literal["manhattan", "euclidean", "chebyshev"]] = [
        "manhattan",
        "euclidean",
        "chebyshev",
    ]

    kvalues = [i for i in range(1, k_max + 1, lag)]
    best_model: KNeighborsClassifier | None = None
    best_params = {"name": "KNN", "metric": metric, "params": ()}
    best_performance = 0.0

    values: dict[str, list] = {}
    for d in dist:
        y_tst_values = []
        for k in kvalues:
            clf = KNeighborsClassifier(n_neighbors=k, metric=d)
            clf.fit(trnX, trnY)
            prdY = clf.predict(tstX)
            val = CLASS_EVAL_METRICS[metric](tstY, prdY)
            y_tst_values.append(val)
            if val - best_performance > DELTA_IMPROVE:
                best_performance = val
                best_params["params"] = (k, d)
                best_model = clf
        values[d] = y_tst_values

    print(f'KNN best with k={best_params["params"][0]} and {best_params["params"][1]}')

    plot_multiline_chart(
        kvalues,
        values,
        title=f"KNN Models ({metric})",
        xlabel="k",
        ylabel=metric,
        percentage=True,
    )

    return best_model, best_params

def trees_study(
    trnX: ndarray,
    trnY: array,
    tstX: ndarray,
    tstY: array,
    d_max: int = 10,
    lag: int = 2,
    metric: str = "accuracy",
):
    criteria = ["entropy", "gini"]
    depths = [i for i in range(2, d_max + 1, lag)]

    best_model = None
    best_params = {"name": "DT", "metric": metric, "params": ()}
    best_performance = 0.0

    values = {}
    for c in criteria:
        y_tst_values = []
        for d in depths:
            clf = DecisionTreeClassifier(
                max_depth=d,
                criterion=c,
                min_impurity_decrease=0,
                random_state=42,
            )
            clf.fit(trnX, trnY)
            prdY = clf.predict(tstX)
            val = CLASS_EVAL_METRICS[metric](tstY, prdY)
            y_tst_values.append(val)
            if val - best_performance > DELTA_IMPROVE:
                best_performance = val
                best_params["params"] = (c, d)
                best_model = clf
        values[c] = y_tst_values

    print(f'DT best with {best_params["params"][0]} and d={best_params["params"][1]}')

    plot_multiline_chart(
        depths,
        values,
        title=f"DT Models ({metric})",
        xlabel="d",
        ylabel=metric,
        percentage=True,
    )

    return best_model, best_params

def mlp_study(
    trnX: ndarray,
    trnY: array,
    tstX: ndarray,
    tstY: array,
    nr_max_iterations: int = 400,
    lag: int = 100,
    metric: str = "accuracy",
):
    nr_iterations = [lag] + [i for i in range(2 * lag, nr_max_iterations + 1, lag)]

    lr_types: list[Literal["constant", "invscaling", "adaptive"]] = [
        "constant",
        "invscaling",
        "adaptive",
    ]
    learning_rates = [0.5, 0.05, 0.005, 0.005]

    best_model: MLPClassifier | None = None
    best_params = {"name": "MLP", "metric": metric, "params": ()}
    best_performance = 0.0

    _, axs = subplots(1, len(lr_types), figsize=(len(lr_types) * HEIGHT, HEIGHT), squeeze=False)

    for i, lr_type in enumerate(lr_types):
        values = {}
        for lr in learning_rates:
            warm_start = False
            y_tst_values = []
            for _ in range(len(nr_iterations)):
                clf = MLPClassifier(
                    learning_rate=lr_type,
                    learning_rate_init=lr,
                    max_iter=lag,
                    warm_start=warm_start,
                    activation="logistic",
                    solver="sgd",
                    verbose=False,
                )
                clf.fit(trnX, trnY)
                prdY = clf.predict(tstX)
                val = CLASS_EVAL_METRICS[metric](tstY, prdY)
                y_tst_values.append(val)
                warm_start = True
                if val - best_performance > DELTA_IMPROVE:
                    best_performance = val
                    best_params["params"] = (lr_type, lr, nr_iterations[len(y_tst_values) - 1])
                    best_model = clf
            values[lr] = y_tst_values

        plot_multiline_chart(
            nr_iterations,
            values,
            ax=axs[0, i],
            title=f"MLP with {lr_type}",
            xlabel="nr iterations",
            ylabel=metric,
            percentage=True,
        )

    print(
        f'MLP best for {best_params["params"][2]} iterations '
        f'(lr_type={best_params["params"][0]} and lr={best_params["params"][1]}) '
        f'with {metric}={best_performance:.6f}'
    )

    return best_model, best_params

def separate_train_test(
    data: DataFrame, 
    target: str, 
    test_size: float = 0.3,
    random_state: int = 42
) -> {DataFrame, DataFrame, DataFrame, DataFrame}:
    df = data.copy()
    
    y = df.pop(target).values
    X = df.values
    
    # Split the data
    trnX, tstX, trnY, tstY = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return trnX, tstX, trnY, tstY

def evaluate_approach(
    data: DataFrame, 
    target: str = "class", 
    metric: str = "recall",
    test_size: float = 0.3,
    random_state: int = 42
) -> dict[str, list]:
    df = data.copy()
    
    y = df.pop(target).values
    X = df.values
    
    # Split the data
    trnX, tstX, trnY, tstY = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    eval: dict[str, list] = {}

    eval_NB: dict[str, float] = run_NB(trnX, trnY, tstX, tstY, metric=metric)
    eval_KNN: dict[str, float] = run_KNN(trnX, trnY, tstX, tstY, metric=metric)
    
    if eval_NB != {} and eval_KNN != {}:
        for met in CLASS_EVAL_METRICS:
            eval[met] = [eval_NB[met], eval_KNN[met]]
        eval["confusion_matrix"] = [eval_NB["confusion_matrix"], eval_KNN["confusion_matrix"]]
    
    return eval

def evaluate_and_plot(
    df: DataFrame,
    lab_folder: str,
    file_tag: str,
    approach: str,
    target_name: str = "class",
    metric: str = "recall"
) -> None:
    figure()
    eval: dict[str, list] = evaluate_approach(data=df, target=target_name, metric=metric)
    plot_multibar_chart(
        ["NB", "KNN"], {k: v for k, v in eval.items() if k != "confusion_matrix"}, title=f"{file_tag}-{approach} evaluation", percentage=True
    )
    savefig(f"../../charts/{lab_folder}/{file_tag}_{approach}_nb_vs_knn_performance.png", bbox_inches='tight')
    show()

    fig, axs = subplots(1, 2, figsize=(2 * HEIGHT, HEIGHT))
    labels = df[target_name].unique()
    labels.sort()
    plot_confusion_matrix(eval["confusion_matrix"][0], labels, ax=axs[0])
    axs[0].set_title(f"{file_tag}-{approach} NB Confusion Matrix")
    plot_confusion_matrix(eval["confusion_matrix"][1], labels, ax=axs[1])
    axs[1].set_title(f"{file_tag}-{approach} KNN Confusion Matrix")
    savefig(f"../../charts/{lab_folder}/{file_tag}_{approach}_nb_vs_knn_confusion_matrix.png", bbox_inches='tight')
    show()
