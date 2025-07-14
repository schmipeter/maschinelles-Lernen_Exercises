#!/home/pekandolf/.local/bin/pdm run
import model
from data import load_cats_vs_dogs
from myio import save_skops as save
import logging
from pathlib import Path
from omegaconf import OmegaConf
from dvclive import Live


def param_from_yaml(live, component):
    prefix = component.type.split(".")[-1]
    for name, value in component.init_args.items():
        live.log_param(prefix + "/" + name, value)


logging.basicConfig(format="%(levelname)s:%(name)s: %(message)s", level=logging.DEBUG)
X_train, y_train, X_test, y_test = load_cats_vs_dogs()

logging.debug("Load config")
params_path = Path(".") / "params.yaml"
params = OmegaConf.load(params_path)

components = [params["PCA"], params["VotingClassifier"]]
estimators = [params[i] for i in components[1].estimators]
voting_clf = model.get(components, estimators)

with Live(dir=Path(".") / "live") as live:

    param_from_yaml(live, components[0])
    for est in estimators:
        param_from_yaml(live, est)

    logging.debug("Train classifier")
    voting_clf.fit(X_train, y_train)

    metrics = model.evaluate(
        voting_clf, [[X_train, y_train, "train"], [X_test, y_test, "test"]]
    )
    for metric_name, value in metrics.items():
        live.log_metric(metric_name, value)

    y_pred = voting_clf.predict(X_test)
    live.log_sklearn_plot("confusion_matrix", y_test, y_pred)

    dir = Path(live.dir) / "artifacts"
    dir.mkdir(parents=True, exist_ok=True)

    file = dir / "model.skops"
    save(voting_clf, file, X_train[:1])
    live.log_artifact(file.relative_to(Path(".")), type="model")
