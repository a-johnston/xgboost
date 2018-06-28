# coding: utf-8
from __future__ import absolute_import

from pyspark import SparkContext
from pyspark.ml.util import JavaMLReadable
from pyspark.ml.util import JavaMLReader
from pyspark.ml.util import JavaMLWritable
from pyspark.ml.wrapper import JavaEstimator
from pyspark.ml.wrapper import JavaModel

__all__ = ['XGBoost', 'XGBoostModel']


def _update_params(sc, params):
    params = dict(params or {})

    max_nthread = int(sc.getConf().get('spark.task.cpus', '1'))
    max_cores = int(sc.getConf().get('spark.cores.max'))
    exec_cores = int(sc.getConf().get('spark.executor.cores'))

    expected_executors = int(max_cores / exec_cores)
    expected_workers = expected_executors * int(exec_cores / max_nthread)

    params.setdefault('nthread', max_nthread)
    params.setdefault('nworkers', expected_workers)
    params.setdefault('num_round', 10)  # Default from the python api

    # For compatibility with https://github.com/dmlc/xgboost/pull/3387
    params.setdefault('num_workers', params['nworkers'])
    if 'train_test_ratio' in params:
        params['trainTestRatio'] = params['train_test_ratio']
    if 'num_early_stopping_rounds' in params:
        params['numEarlyStoppingRounds'] = params['num_early_stopping_rounds']

    return params


class XGBoost(JavaEstimator, JavaMLReadable, JavaMLWritable):
    """PySpark estimator for XGBoost models.
    """

    def __init__(self, params=None):
        """Runs distributed XGBoost training on the provided dataframe and parameters. If not provided
        in `params`, sets nthread and workers based on current spark config. The standard set of
        options can be found: https://xgboost.readthedocs.io/en/latest/parameter.html

        By default, will use columns 'label' and 'features' to read each labeled feature vector, where
        the feature column contains pyspark ml.linalg.Vector instances.

        Common parameters (default):
        - num_round : The number of rounds for boosting (10)
        - max_depth : The max depth to grow a single tree (6)
        - objective : Objective function for training (reg:linear)
            Example options: reg:logistic, rank:pairwise

        Additionally, xgboost4j supports the following options:
        - train_test_ratio: Percent of data to use for training.
        - num_early_stopping_rounds: Rounds to terminate after following no improvement to metric.

        >>> params = { .. }
        >>> model = xgb.XGBoost(params).fit(data)
        >>> model
        XGBoostRegressionModel_d76bf6ebdb03

        :param dict[str: object] params: Map of XGBoost parameters.
        """
        super(XGBoost, self).__init__()
        sc = SparkContext.getOrCreate()
        params = _update_params(sc, params)

        self._paramMap = params
        self._java_obj = self._new_java_obj(
            self._java_classname(),
            sc._jvm.PythonUtils.toScalaMap(params),
        )

    @classmethod
    def read(cls):
        return XGBoostClassReader(cls)

    def _create_model(self, java_model):
        return XGBoostModel(java_model)

    @classmethod
    def _java_classname(cls):
        return 'ml.dmlc.xgboost4j.scala.spark.XGBoostEstimator'


class XGBoostModel(JavaModel, JavaMLReadable, JavaMLWritable):
    """PySpark XGBoost model.
    """

    def save_booster(self, path):
        """Saves just the booster represented by the model. This booster can subsequently be loaded
        in the python xgboost api with xgb.Booster(model_file=path). The booster can also be
        reloaded with XGBoostModel.load_booster(path).

        :param str path: Path to save model to.
        """
        if 'booster' in dir(self._java_obj):
            self._java_obj.booster().saveModel(path)
        else:
            # Compatible with https://github.com/dmlc/xgboost/pull/3387
            self._java_obj._booster().saveModel(path)

    @classmethod
    def load_booster(cls, path):
        """Loads an XGBoostModel instance from the booster at the specified filename.

        :param str path: File to load booster instance from.
        :return XGBoostModel: Instance which wraps the loaded booster for distributed use.
        """
        sc = SparkContext.getOrCreate()
        java_booster = sc._jvm.ml.dmlc.xgboost4j.java.XGBoost.loadModel(path)
        scala_booster = sc._jvm.ml.dmlc.xgboost4j.scala.Booster(java_booster)
        return cls(JavaModel._new_java_obj(cls._java_classname(), cls._randomUID(), scala_booster))

    @property
    def training_summary(self):
        """Returns a dictionary of each evaluation set's objective metric at each round. If the
        test/train ratio is 1.0, only train history is returned. Does not return watch results.
        The metric used is specified by the parameter 'eval_metric' defaulting to:
        - RMSE for regression (used by the default objective of 'reg:linear')
        - error for classification
        - MAP for ranking
        Other options include 'logloss', 'auc', and 'ndcg'.
        https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters

        >>> model = XGboost({'train_test_ratio': 0.8}).fit(data)
        >>> model.objective_history['train']
        [0.358666, 0.265099, ...]
        >>> model.objective_history['test']
        [0.358689, 0.265152, ...]

        :return dict[str: list[int]]: Objective history for each evaluation set.
        """
        summary = self._java_obj.summary()
        results = {'train': list(summary.trainObjectiveHistory())}
        if summary.testObjectiveHistory().nonEmpty():
            results['test'] = list(summary.testObjectiveHistory().get())
        return results

    @classmethod
    def read(cls):
        return XGBoostClassReader(cls)

    @classmethod
    def _from_java(cls, java_model):
        return XGBoostModel(java_model)

    @classmethod
    def _java_classname(cls):
        return 'ml.dmlc.xgboost4j.scala.spark.XGBoostRegressionModel'


class XGBoostClassReader(JavaMLReader):
    @classmethod
    def _java_loader_class(cls, clazz):
        return clazz._java_classname()


# The following classes are for compatibility with https://github.com/dmlc/xgboost/pull/3387


class XGBoostRegressor(XGBoost):
    @classmethod
    def _java_classname(cls):
        return 'ml.dmlc.xgboost4j.scala.spark.XGBoostRegressor'

    def _create_model(self, java_model):
        return XGBoostRegressionModel(java_model)


class XGBoostRegressionModel(XGBoostModel):
    pass


class XGBoostClassifier(XGBoost):
    @classmethod
    def _java_classname(cls):
        return 'ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier'

    def _create_model(self, java_model):
        return XGBoostClassificationModel(java_model)


class XGBoostClassificationModel(XGBoostModel):
    @classmethod
    def _java_classname(cls):
        return 'ml.dmlc.xgboost4j.scala.spark.XGBoostClassificationModel'
