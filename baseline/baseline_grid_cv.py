# baseline_grid_cv.py
#
# Author: Ekaterina Kondrateva, 2020
# Updated constantly in@neurobot package, check out https://github.com/kondratevakate/neurobot

"""
This script provides pipeline for GridSearch Cros-Validation with incorporated
data dimensionality reduction and feature selection. There prescribed methods of model validation, including LOO and Bootstrapping, that make the whole pipeline plug-and-play solution for small sample data analysis.

Ready to use for ML baseline for Neuroimaging features classification.

Usage:
    temp_grid = GridCV(X, y)
    grid_schz_vs_c = temp_grid.train()
"""


# general imports
import glob, os
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import pickle
import json
from copy import deepcopy

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
sns.set_style('darkgrid')

# warnings
import warnings
warnings.filterwarnings("ignore")

# sklearn
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, StratifiedShuffleSplit, cross_val_score, cross_val_predict, GridSearchCV, LeaveOneOut
from sklearn.feature_selection import SelectFromModel, VarianceThreshold, SelectKBest, f_classif, chi2
from sklearn.feature_selection.base import SelectorMixin
from sklearn.feature_selection.from_model import _get_feature_importances

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer as Imputer
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding

### metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, confusion_matrix

### utils 
from sklearn.pipeline import Pipeline
import joblib
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.utils.metaestimators import if_delegate_has_method

# others
from scipy import stats
from mlxtend.evaluate import bootstrap_point632_score
from collections import Counter, defaultdict
from datetime import date




class SelectNFeaturesFromModel(BaseEstimator, SelectorMixin, MetaEstimatorMixin):
    import warnings
    warnings.filterwarnings("ignore")

    def __init__(self, estimator, n_selected, prefit=False):
        self.estimator = estimator
        self.n_selected = n_selected
        self.prefit = prefit

    def _get_support_mask(self):
        if self.prefit:
            estimator = self.estimator
        elif hasattr(self, 'estimator_'):
            estimator = self.estimator_
        else:
            raise ValueError(
                'Either fit SelectFromModel before transform or set "prefit='
                'True" and pass a fitted estimator to the constructor.')
        scores = _get_feature_importances(estimator)
        threshold = np.sort(scores)[-self.n_selected]
        return scores >= threshold

    def fit(self, X, y=None, **fit_params):
        if self.prefit:
            raise NotFittedError(
                "Since 'prefit=True', call transform directly")
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y, **fit_params)
        return self
    
    @property
    def scores_(self):
        scores = _get_feature_importances(self.estimator_,)
        return scores

    @property
    def threshold_(self):
        scores = _get_feature_importances(self.estimator_,)
        return np.sort(scores)[-n_selected]
    
    @if_delegate_has_method('estimator')
    def partial_fit(self, X, y=None, **fit_params):
        if self.prefit:
            raise NotFittedError(
                "Since 'prefit=True', call transform directly")
        if not hasattr(self, "estimator_"):
            self.estimator_ = clone(self.estimator)
        self.estimator_.partial_fit(X, y, **fit_params)
        return self


##########################################  FUNCTIONS  ##################################################
##########################################FUNCTIONS  ########################### 
    
def get_svc_grid(cv, dim_reduction_methods, scoring, random_state=None, n_jobs=1,
                 svc_kernel_l=None, svc_c_l=None, svc_gamma_l=None):
    
    get_svc_grid.__doc__ = "A function returning pre-defined pipeline for svc binary classification"

    pipe = Pipeline([
        ("Fill_NaN", Imputer(strategy="median")),
        ('StdScaler', StandardScaler()),
        ('dim_reduction', SelectKBest(stats.ttest_ind)),
        ('classifier', SVC(probability=True, random_state=random_state)),])

    param_grid = {'dim_reduction': dim_reduction_methods,}
    if svc_kernel_l is not None:
        param_grid['classifier__kernel'] = svc_kernel_l
    if svc_c_l is not None:
        param_grid['classifier__C'] = svc_c_l
    if svc_gamma_l is not None:
        param_grid['classifier__gamma'] = svc_gamma_l
    
    return GridSearchCV(
        estimator = pipe, param_grid = param_grid,
        scoring = scoring, cv = cv, n_jobs = n_jobs
    )


def get_lr_grid(cv, dim_reduction_methods, scoring, random_state = None, n_jobs = 1,
                 lr_c_l = None, lr_penalty_l = None):
    
    pipe = Pipeline(
        [
            ("Fill_NaN", Imputer(strategy="median")),
            ('StdScaler', StandardScaler()),
            ('dim_reduction', SelectKBest(stats.ttest_ind)),
            ('classifier', LogisticRegression(random_state = random_state)),
        ]
    )

    param_grid = {'dim_reduction': dim_reduction_methods,}
    if lr_c_l is not None:
        param_grid['classifier__C'] = lr_c_l
    if lr_penalty_l is not None:
        param_grid['classifier__penalty'] = lr_penalty_l
    
    return GridSearchCV(
        estimator = pipe, param_grid = param_grid,
        scoring = scoring, cv = cv,
        n_jobs = n_jobs
    )


def get_rfc_grid(cv, dim_reduction_methods, scoring, random_state=None, n_jobs=1, 
                 rfc_n_estimators_l=None):

    pipe = Pipeline([
        ("Fill_NaN", Imputer(strategy="median")),
        ('StdScaler', StandardScaler()),
        ('dim_reduction', SelectKBest(stats.ttest_ind)),
        ('classifier', RandomForestClassifier(random_state=random_state)),])

    param_grid = {'dim_reduction': dim_reduction_methods,}
    if rfc_n_estimators_l is not None:
        param_grid['classifier__n_estimators'] = rfc_n_estimators_l
    
    return GridSearchCV(
        estimator = pipe, param_grid = param_grid,
        scoring = scoring, cv = cv, n_jobs = n_jobs
    )


def get_knn_grid(cv, dim_reduction_methods, scoring, random_state = None, n_jobs = 1,
                 knn_n_neighbors_l = None, knn_weights_l = None, knn_p_l = None):
    
    pipe = Pipeline([
        ("Fill_NaN", Imputer(strategy = "median")),
        ('StdScaler', StandardScaler()),
        ('dim_reduction', SelectKBest(stats.ttest_ind)),
        ('classifier', KNeighborsClassifier()),])

    param_grid = {'dim_reduction': dim_reduction_methods,}
    if knn_n_neighbors_l is not None:
        param_grid['classifier__n_neighbors'] = knn_n_neighbors_l
    if knn_weights_l is not None:
        param_grid['classifier__weights'] = knn_weights_l
    if knn_p_l is not None:
        param_grid['classifier__p'] = knn_p_l
    
    return GridSearchCV(
        estimator = pipe, param_grid = param_grid,
        scoring = scoring, cv = cv, n_jobs = n_jobs
    )
    
    
def repeated_cross_val_predict_proba(estimator, X, y, n_objects, 
                                     cv, pos_label = None, file = None):
    if pos_label is None:
        y_enc = pd.Series(LabelEncoder().fit_transform(y), index = y.index)
    else:
        y_enc = pd.Series(y == pos_label, dtype = int)
    predictions = [[] for i in range(n_objects)]
    for idx_tr, idx_te in tqdm(cv.split(X, y_enc)):
        estimator.fit(X.iloc[idx_tr], y_enc.iloc[idx_tr])
        pred_te = np.array(estimator.predict_proba(X.iloc[idx_te]), dtype = float)
        for i, idx in enumerate(idx_te):
            predictions[X.index[idx]].append(pred_te[i, 1])
        
    predictions = pd.DataFrame(predictions)
    if file is not None:
        predictions.to_csv(file)
    return predictions


def get_feature_sets_on_cross_val(X, y, model, cv):
    feature_sets = []
    for idx_tr, idx_te in tqdm(cv.split(X, y)):
        X_tr = X.loc[X.index[idx_tr]]
        y_tr = y.loc[X.index[idx_tr]]
        y_te = y.loc[X.index[idx_te]]
        feature_sets.append(get_features(X_tr, y_tr, model))
    return feature_sets


def get_features(X, y, model):
    model.fit(X, y)
    dim_reduction = model.named_steps["dim_reduction"]
    features_idx = dim_reduction.get_support()
    features = X.columns[features_idx].tolist()
    return features


def plot_roc_curve(y, probas, average_repeats=False, show=True):
    if average_repeats:
        y_true = y
        y_score = probas.mean(axis=1)
    else:
        n_repeats = probas.shape[1]
        y_true = pd.Series(np.tile(y, (n_repeats)), dtype=int)
        y_score = probas.values.T.reshape(-1, 1)
    fpr, tpr, t = roc_curve(y_true=y_true, y_score=y_score)
    
    if show:
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive rate", fontsize=14)
        plt.ylabel("True Positive rate", fontsize=14)
        plt.show()
        print("auc =", roc_auc_score(y_true, y_score))        
    return fpr, tpr, t


def get_fpr_fnr(fpr, tpr, fix_fpr_l=[0.1, 0.15, 0.2, 0.3]):
    fnr_l = []
    for fix_fpr in fix_fpr_l:
        fnr_l.append(1 - tpr[fpr <= fix_fpr][-1])
    fpr_fnr_table = pd.DataFrame(
        np.column_stack((fix_fpr_l, fnr_l)), columns=["False Positive rate (fixed)", "False Negative rate"]
    )
    display(fpr_fnr_table)
    return fpr_fnr_table


# TODO: сейчас работает только для одномерных векторов вероятностей 
# (одно предсказание для каждого объекта, напр. leave one out)
def get_incorrectly_classified(labels, y, probas, idx, fpr, t, fix_fpr_l=[0.1, 0.15, 0.2, 0.3], file=None, show=True):
    columns = ["False Positive rate (fixed)", "Threshold", "False Positives indexes", "False Negatives indexes"]
    t_l = []
    false_0 = []
    false_1 = []
    for fix_fpr in fix_fpr_l:
        fix_t = t[fpr <= fix_fpr][-1]
        t_l.append(fix_t)
        labels_t = probas > fix_t
        labels_t = pd.Series(labels_t.values.ravel())
        false_0.append(
            ", ".join(
                list(
                    labels.loc[(probas[idx][np.logical_and(labels_t[idx] == 0, y == 1)]).index, "patient_number"]
                )
            )
        )
        false_1.append(
            ", ".join(
                list(
                    labels.loc[(probas[idx][np.logical_and(labels_t[idx] == 1, y == 0)]).index, "patient_number"]
                )
            )
        )
              
    t_l = np.array(t_l)
    false_0 = np.array(false_0)
    false_1 = np.array(false_1)
    
    res = pd.DataFrame(np.column_stack((fix_fpr_l, t_l)), columns=columns[:2])
    res["False Positives indexes"] = false_1
    res["False Negatives indexes"] = false_0
    
    if file is not None:
        res.to_csv(file)
        
    if show:
        display(res)
    return res


def print_results_(clf_grid_dict, save_plot_to=None):
    results = {
            "classifier" : [], 
            "best parameters" : [],
            "best dim. reduction method" : [],
            "mean" : [], 
            "std" : []
           }
    
    for clf, grid in clf_grid_dict.items():
        results["classifier"].append(clf)
        results["best parameters"].append(
            ", ".join(
                [param + " = " + str(best_value) for param, best_value in grid.best_params_.items() if param != 'dim_reduction']
            )
        )
        results["best dim. reduction method"].append(grid.best_params_['dim_reduction'])
        idx = grid.best_index_    # это вроде лучший score среди 5x5=25 измерений
        results["mean"].append(grid.cv_results_['mean_test_score'][idx])   # значит, здесь можно аппендить mean
        results["std"].append(grid.cv_results_['std_test_score'][idx])
        
    results = pd.DataFrame(
        results, columns=["classifier", "best parameters", "best dim. reduction method", "mean", "std"]
    )
    display(results.set_index("classifier"))
    
    # draw graph
    width = 0.9
    for i in results.index:
        plt.bar(i, results.loc[i, "mean"], width, yerr=results.loc[i, "std"], label=results.loc[i, "classifier"])
    plt.xticks(range(results.shape[0]), results.loc[:, "classifier"])
    plt.axis(ymin=0.0, ymax=1.0)
    if save_plot_to is not None:
        plt.savefig(save_plot_to)
    plt.show()
    
    print("Best model: ")
    clf = results.loc[results["mean"].argmax(), "classifier"]
    print(clf)
    print("\n".join(
            [param + " = " + str(best_value) for param, best_value in clf_grid_dict[clf].best_params_.items()]))

##############################################  MAIN CLASS  ##################################################
    
class GridCV:
    
    """
    A class used to search among several classifiers with different assesement

    ...

    Attributes
    ----------
    X : pandas.DataFrame
        The training data
    y : pandas.DataFrame
        The target to training data
    problem_name : str, optional
        Classificator name for saving model and meta- files
    
    Methods
    -------
    train()
        Performs the grid search among classifiers
    save_best_models(path='')
        Saves best models to dedicated path
    print_results()
        Displays the best models with hyperparameters chosen
    
    """
    
   
    def __init__(self, X, y, 
                 problem_name = 'test_classification',
                 n_splits = 5, 
                 n_repeats = 5,
                 scoring = 'roc_auc',
                 random_state = 42,
                 n_jobs = -1,
#                  grid = []
                ):
        """
        Parameters
        ----------
        X : pandas.DataFrame
            The training data
        y : pandas.DataFrame
            The target to training data
        problem_name : int, optional
            Classificator name for saving model and meta- files
        """
        
        
        self.X = X
        self.y = y
        self.problem_name = problem_name
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.scoring = scoring
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.pos_label = None
        self.features_groups = []
        self.save_plot_to = None
        self.path = ''
        self.n_objects = self.X.shape[0]
        self.classifiers = ['best', "svc", "lr", "rfc", "knn"]
        self.grid = []
        self.loo_results = [] #массив результатов loo для сохранения в json
        self.bootstrap_results = [] #массив результатов bootstrap для сохранения в json
        self.bootstrap_boundaries = [] #массив {min; max} результатов bootstrap для построения графика
        
    def train(self):
        """ Performs the grid search among classifiers
        
        """
        
        print ('Number of samples ', self.X.shape[0], "\n")
        print ('Number of features ', self.X.shape[1], "\n")
        
        cv = RepeatedStratifiedKFold(
            n_splits = self.n_splits,
            n_repeats = self.n_repeats,
            random_state = self.random_state,
        )
        
        n_features = [100]
        n_components = [20] # 40 
        
        dim_reduction_methods = []
#         dim_reduction_methods += [SelectKBest(stats.ttest_ind, n) for n in n_features] #chi2 или убрать
        dim_reduction_methods += [SelectKBest(f_classif, n) for n in n_features]
        dim_reduction_methods += [SelectNFeaturesFromModel(
            RandomForestClassifier(
                n_estimators = 100,
                random_state = self.random_state), n
        ) for n in n_features]
        
        dim_reduction_methods += [SelectNFeaturesFromModel(
            LogisticRegression(
                random_state = self.random_state), n
        ) for n in n_features]
        
#         dim_reduction_methods += [SelectNFeaturesFromModel(
#             ExtraTreesClassifier(
#                 n_estimators = 100,
#                 random_state = self.random_state), n
#         ) for n in n_features]

        print("Target distribution: ")
        print(self.y.value_counts(), "\n")
        
        if self.pos_label is None: 
            y_enc = pd.Series(
                LabelEncoder().fit_transform(self.y), index = self.y.index
            )
        else: 
            y_enc = pd.Series(
                self.y == pos_label, dtype=int
            )


        print("Training SVC...")
        grid_cv_svc = get_svc_grid(
            cv, dim_reduction_methods, self.scoring, 
            random_state = self.random_state,
            n_jobs = self.n_jobs,
            svc_kernel_l = ["rbf", "linear"],
#             svc_c_l = [10 ** i for i in range(1, 4, 1)],
#             svc_gamma_l = [10 ** i for i in range(-3, -1, 1)]
        )
        start_time = time.time()
        grid_cv_svc.fit(self.X, y_enc)
        print("(training took {}s)\n".format(time.time() - start_time))


        print("Training LR...")
        grid_cv_lr = get_lr_grid(
            cv, dim_reduction_methods, self.scoring,
            random_state = self.random_state, 
            n_jobs = self.n_jobs,
            lr_c_l = [10 ** i for i in range(-4, -1, 1)],
            lr_penalty_l = ["l1", "l2"]
        )
        start_time = time.time()
        grid_cv_lr.fit(self.X, y_enc)
        print("(training took {}s)\n".format(time.time() - start_time))
        

        print("Training RFC...")
        grid_cv_rfc = get_rfc_grid(
            cv, dim_reduction_methods, self.scoring,
            random_state = self.random_state, n_jobs=self.n_jobs,
            rfc_n_estimators_l = [i for i in range(100, 210, 30)]
        )
        start_time = time.time()
        grid_cv_rfc.fit(self.X, y_enc)
        print("(training took {}s)\n".format(time.time() - start_time))
        
        
        print("Training KNN...")
        class_size_tr = min(self.y.value_counts())
        grid_cv_knn = get_knn_grid(
            cv, dim_reduction_methods, self.scoring,
            random_state = self.random_state, 
            n_jobs = self.n_jobs,
#             knn_p_l = [1, 2], 
#             knn_weights_l = ["uniform", "distance"],
#             knn_n_neighbors_l = [i for i in range(5, class_size_tr - 1, 3)]
        )

        start_time = time.time()
        grid_cv_knn.fit(self.X, y_enc)
        print("(training took {}s)".format(time.time() - start_time))
        
        print("Scoring:", self.scoring)

        best_model = max(
            [grid_cv_svc, grid_cv_lr, grid_cv_rfc, grid_cv_knn], key=lambda x: x.best_score_
        ).best_estimator_
        self.grid = [best_model, grid_cv_svc, grid_cv_lr, grid_cv_rfc, grid_cv_knn]
        return self.grid
    
    
    def save_best_models(self, path=''):
        dirName = path + "{}_bests/{}_bests/".format(self.mri_data.shape[0], self.datatype)
        if not os.path.exists(dirName):
            os.makedirs(dirName)
        for i, clf in enumerate(self.classifiers):
            model = self.grid[i]
            with open(dirName + "{}_{}_best_{}.pkl".format(self.datatype, self.problem_name, clf.upper()), "wb") as file:
                pickle.dump(model, file)
            print (clf.upper(), ' saved')        
        print ('DONE')
        print ('Path of files:', (dirName))

# TODO: write model loader       
#             with open(dirName + "{}_bests/{}_{}_best_{}.pkl".format(self.datatype, self.datatype,
#                                                         self.problem_name, clf.upper()), "rb") as file:
#                 loaded_model = pickle.load(file)

            
    def print_results(self):
        print_results_(
            {
                "SVC" : self.grid[1],
                "LR" : self.grid[2],
                "RFC" : self.grid[3],
                "KNN" : self.grid[4],
            }
        )
    
    
    def print_metrics(self):
        grid_cv = self.grid[1:]
        best_model_l = [x.best_estimator_ for x in grid_cv]
        classifiers_l = self.classifiers[1:]
        probas_l = [
            repeated_cross_val_predict_proba(
                best_model, self.X, self.y, self.n_objects,
                cv=LeaveOneOut(), 
                file="{}_probas_mri_{}.csv".format(
                classifiers_l[i],
                self.problem_name)
            ) for i, best_model in enumerate(best_model_l)
        ]
        feature_sets_l = [
            get_feature_sets_on_cross_val(
                self.X, self.y, best_model, cv=LeaveOneOut()
            ) for best_model in best_model_l
        ]
        features_l = [
            pd.DataFrame(
                pd.DataFrame(data=feature_set).stack().reset_index(drop=True).value_counts(),
                columns=['frequency']
            ) for feature_set in feature_sets_l]

        for i in range(len(best_model_l)):
            print(best_model_l[i])
            fpr, tpr, t = plot_roc_curve(self.y, probas_l[i], self.idx)
            get_fpr_fnr(fpr, tpr)
            get_incorrectly_classified(self.labels, self.y, probas_l[i], self.idx, fpr, t)
            display(features_l[i].iloc[:10])
            
            
    def loo_cv(self):
        """Performs Leave-One-Out cross validation.
        """

        print (self.problem_name)
        for k in range(1, len(self.grid)):
            start_time = time.time()
            best_model = self.grid[k].best_estimator_
            loo = LeaveOneOut()
            loo.get_n_splits(self.X)
            predict = []
            
            for train_index, test_index in loo.split(self.X):
                X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
                y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
                predict.append(best_model.fit(X_train, y_train).predict(X_test)[0])
                
            tpr, fpr, fnr, tnr = (confusion_matrix(self.y, 
                                                   predict
                                                   ).astype('float') 
                                  /confusion_matrix(self.y, 
                                                    predict
                                                    ).sum(axis=1)[:, np.newaxis]
                                 ).ravel()
            end_time = np.round(time.time() - start_time, 2)
            self.loo_results.append([np.round(((tpr + tnr)*100) / 2, 2), 
                                     np.round(tpr*100, 2), np.round(tnr*100, 2), 
                                     end_time])
            print(
                self.classifiers[k].upper()  + ': ',
                ' acc', np.round((tpr + tnr) / 2, 2), 
                ' tpr', np.round(tpr, 2),
                ' tnr', np.round(tnr, 2),
                ' time', end_time
            )



    def bootstrap_632(self):
        """Performs bootstrap validation.
        """

        print (self.problem_name)
        for k in range(1, len(self.grid)):
            start_time = time.time()
            scores = bootstrap_point632_score(
                self.grid[k].best_estimator_, self.X.values,
                self.y.values, n_splits=1000,
                method='.632', random_seed=42
            )
            acc = np.mean(scores)
            lower = np.percentile(scores, 2.5)
            upper = np.percentile(scores, 97.5)
            end_time = np.round(time.time() - start_time, 2)
            self.bootstrap_results.append([np.round(100*acc, 2), 
                                           [np.round(100*lower, 2), 
                                            np.round(100*upper, 2)], 
                                           end_time])
            print(
                self.classifiers[k].upper(),
                ' acc: %.2f%%' % (100*acc), 
                ' 95%% Confidence interval: [%.2f, %.2f]' % \
                    (100*lower, 100*upper),
                ' time', end_time
            )
        
        
    def plot_val(self, save_fig=True, fig_name='val'):
        """Plots validation results and saves the figure.

        Parameters
        ----------
        fig_name : str
            Name of the figure to be saved.
        """
        
        fig, (ax1, ax2) = plt.subplots(2, figsize=(12,12), 
                                       sharex=True, sharey=True)
        ax1.set_title('LeaveOneOut validation')
        for i in range(len(self.bootstrap_results)):
            acc, = ax1.plot(i, self.bootstrap_results[i][0], "o", 
                            color="pink", label = 'acc')
            conf = ax1.vlines(i, self.bootstrap_results[i][1][0], 
                              self.bootstrap_results[i][1][1], 
                              colors='k', linestyles='solid', label='conf int')
        labels = ['acc', 'conf. int.']
        ax1.legend(labels = labels, handles=[acc, conf], loc = 'best')
    
        ax2.set_title('Bootstrap validation')
        for i in range(len(self.loo_results)):
            ax2.plot(i, self.loo_results[i][0], 
                     "o", color="darkred", label="accuracy")    
            ax2.plot(i, self.loo_results[i][1],
                     "*", color="black", label="tnr")
            ax2.plot(i, self.loo_results[i][2], 
                     "*", color="g", label="tpr")
        plt.xticks(range(len(self.loo_results)), 
                   [i.upper() for i in self.classifiers[1:]])
        labels = ['acc', 'tnr', 'tpr']
        ax2.legend(labels, loc = 'best')
        plt.show()
        if save_fig == True:
            fig.savefig(fig_name + '.png')    
        
            
    def save_val_results(self, problem_name = '', path=''):
        """Saves validation results.

        Parameters
        ----------
        problem_name : str
            Dataset name and problem type.
        path : str
            Path to the folder where models should be saved.
        """
        
        tree = lambda: collections.defaultdict(tree)
        model_param = tree()
        for i, clf in enumerate(self.classifiers[1:]):
            model_param['LeaveOneOut'] \
                       [clf.upper()] \
                       ['acc'] = self.loo_results[i][0]
            model_param['LeaveOneOut'] \
                       [clf.upper()] \
                       ['tpr'] = self.loo_results[i][1]
            model_param['LeaveOneOut'] \
                       [clf.upper()] \
                       ['tnr'] = self.loo_results[i][2]
            model_param['LeaveOneOut'] \
                       [clf.upper()] \
                       ['time'] = self.loo_results[i][3]
            model_param['Bootstrap'] \
                       [clf.upper()] \
                       ['acc'] = self.bootstrap_results[i][0]
            model_param['Bootstrap'] \
                       [clf.upper()] \
                       ['Confidence interval'] = self.bootstrap_results[i][1]
            model_param['Bootstrap'] \
                       [clf.upper()] \
                       ['time'] = self.bootstrap_results[i][2]
        json_file = json.dumps(model_param, indent=4)
        today = date.today().strftime("%d%m%Y")
        with open('{}_{}_{}_val_results.json'.format(today, 
                                                     problem_name, 
                                                     self.n_objects), 
                  'w') as file:
            file.write(json_file)
        
        
    def save_models_pkl(self, problem_name='', path=''):
        """Saves models in pkl format.

        Parameters
        ----------
        problem_name : str
            Dataset name and problem type
        path : str
            Path to the folder where models should be saved.
        """
        
        today = date.today().strftime("%d%m%Y")
        for i, clf in enumerate(self.classifiers[1:]):
            model = self.grid[i+1]
            idx = model.best_index_
            mean = model.cv_results_['mean_test_score'][idx]
            std = model.cv_results_['std_test_score'][idx]
            with open(path + "{}_{}_{}_{}_{}_{}.pkl".format(today, problem_name, 
                                                  self.n_objects, 
                                                  clf.upper(), int(mean*100),
                                                  int(std*100)), "wb") as file:
                pickle.dump(model.best_estimator_.fit(self.X, self.y), file)
            
            
    def train_val(self, save=True, val=True, plot=True, save_fig=True, 
                  problem_name='', path='', fig_name='val'):
        """Perfomes training, validation and results printing.
        Saves models and validation results.

        Parameters
        ----------
        problem_name : str
            Dataset name and problem type
        path : str
            Path to the folder where models should be saved.
        fig_name : str
            Name of the figure to be saved.
        """
        
        self.train()
        print('')
        self.print_results()
        if val == True:
            print('\n'+'\033[1m'+'Bootstrap_632:'+'\033[0m')
            self.bootstrap_632()
            print('\n'+'\033[1m'+'Loo_cv:'+'\033[0m')
            self.loo_cv()
        if plot == True:
            self.plot_val(save_fig, fig_name)
        if save == True and val == True :
            print('\n''Saving results...')
            self.save_val_results(problem_name, path)
            self.save_models_pkl(problem_name, path)
            print('Done')
        elif save == True:
            print('\n''Saving results...')
            self.save_models_pkl(problem_name, path)
            print('Done')