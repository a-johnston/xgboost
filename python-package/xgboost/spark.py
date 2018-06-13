# coding: utf-8
from __future__ import absolute_import

from pyspark import SparkContext
from pyspark.ml.util import JavaMLReadable
from pyspark.ml.util import JavaMLReader
from pyspark.ml.util import JavaMLWritable
from pyspark.ml.wrapper import JavaEstimator
from pyspark.ml.wrapper import JavaModel

__all__ = ['XGBoost', 'XGBoostModel']


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
        - trainTestRatio: Percent of data to use for training.
        - numEarlyStoppingRounds: Rounds to terminate after following no improvement to metric.

        >>> path = <S3 path>
        >>> params = { .. }
        >>> data = dataset.LabeledPointFormat().read_dataframe(path)
        >>> model = xgb.XGBoost(params).fit(data)
        >>> model
        XGBoostRegressionModel_d76bf6ebdb03

        :param dict[str: object] params: Map of XGBoost parameters.
        """
        super(XGBoost, self).__init__()
        sc = SparkContext.getOrCreate()
        params = dict(params or {})

        max_nthread = int(sc.getConf().get('spark.task.cpus', '1'))
        max_cores = int(sc.getConf().get('spark.cores.max'))
        exec_cores = int(sc.getConf().get('spark.executor.cores'))

        expected_executors = int(max_cores / exec_cores)
        expected_workers = expected_executors * int(exec_cores / max_nthread)

        params.setdefault('nthread', max_nthread)
        params.setdefault('nworkers', expected_workers)
        params.setdefault('num_round', 10)  # Default from the python api

        self._java_obj = sc._jvm.ml.dmlc.xgboost4j.scala.spark.XGBoostEstimator(
            sc._jvm.PythonUtils.toScalaMap(params),
        )
        self._paramMap = params

    def _create_model(self, java_model):
        return XGBoostModel(java_model)


class XGBoostModel(JavaModel, JavaMLReadable, JavaMLWritable):
    """PySpark XGBoost model.
    """

    def save_booster(self, path):
        """Saves just the booster represented by the model. This booster can subsequently be loaded
        in the python xgboost api with xgb.Booster(model_file=path). The booster can also be
        reloaded with XGBoostModel.load_booster(path).

        :param str path: Path to save model to.
        """
        self._java_obj.booster().saveModel(path)

    @staticmethod
    def load_booster(path, cls='ml.dmlc.xgboost4j.scala.spark.XGBoostRegressionModel'):
        """Loads an XGBoostModel instance from the booster at the specified filename. By default
        loads a regression model but a different class can be provided with the cls kwarg.

        :param str path: File to load booster instance from.
        :return XGBoostModel: Instance which wraps the loaded booster for distributed use.
        """
        sc = SparkContext.getOrCreate()
        java_booster = sc._jvm.ml.dmlc.xgboost4j.java.XGBoost.loadModel(path)
        scala_booster = sc._jvm.ml.dmlc.xgboost4j.scala.Booster(java_booster)
        return XGBoostModel(JavaModel._new_java_obj(cls, scala_booster))

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

        >>> model = XGboost({'trainTestRatio': 0.8}).fit(data)
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
        return XGBoostModelReader(cls)

    @classmethod
    def _from_java(cls, java_model):
        return XGBoostModel(java_model)


class XGBoostModelReader(JavaMLReader):
    # See pyspark.ml.util.JavaMLReader._java_loader_class

    @classmethod
    def _load_java_obj(cls, clazz):
        return SparkContext._jvm.ml.dmlc.xgboost4j.scala.spark.XGBoostModel
