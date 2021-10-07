import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler, LabelBinarizer, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, multilabel_confusion_matrix, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from pyeer.eer_info import get_eer_stats
from pyeer.report import generate_eer_report, export_error_rates
from pyeer.plot import plot_eer_stats
from yellowbrick.classifier import ROCAUC


def division_dataset(x_array, y_array):
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    counter = 0

    for n in range(len(x_array)):
        if counter < 2:
            x_train.append(x_array[n])
        elif 2 <= counter < 5:
            x_test.append(x_array[n])
        elif counter >= 5:
            x_train.append(x_array[n])

        if counter < 2:
            y_train.append(y_array[n])
        elif 2 <= counter < 5:
            y_test.append(y_array[n])
        elif counter >= 5:
            y_train.append(y_array[n])

        if counter == 6:
            counter = 0
        else:
            counter += 1

    return x_train, x_test, y_train, y_test


def DI_calculation(y_test, y_pred, y_pred_prob):
    gen_dis = []
    imp_dis = []

    for x in range(len(y_pred)):
        tt = y_test[x]
        pp = y_pred[x]

        if tt == pp:
            gen_dis.append(y_pred_prob[x][pp - 1])
        else:
            imp_dis.append(y_pred_prob[x][pp - 1])

    norm_gen_dis = gen_dis / np.linalg.norm(gen_dis)
    norm_imp_dis = imp_dis / np.linalg.norm(imp_dis)

    mean_gen = np.mean(norm_gen_dis)
    mean_imp = np.mean(norm_imp_dis)

    std_gen = np.std(norm_gen_dis)
    std_imp = np.std(norm_imp_dis)

    DI = (abs(mean_imp - mean_gen)) / ((((std_gen * std_gen) + (std_imp * std_imp)) / 2) ** 0.5)

    return DI


def plot_roc_curve(model, xtrain, ytrain, xtest, ytest, title, csv_file, figure):
    fig, axes = plt.subplots(1, 1, num=title + '_' + csv_file + '_DD_' + str(figure))
    # Creating visualization with the readable labels
    visualizer = ROCAUC(model, per_class=False, title=title, ax=axes)

    # Fitting to the training data first then scoring with the test data
    visualizer.fit(xtrain, ytrain)
    visualizer.score(xtest, ytest)
    visualizer.show()
    visualizer.finalize()

    return visualizer


# def division_dataset(x_array, y_array):
#     x_train = []
#     x_test = []
#     y_train = []
#     y_test = []
#
#     counter = 0
#
#     for n in range(len(x_array)):
#         if counter < 3:
#             x_train.append(x_array[n])
#         elif 3 <= counter < 8:
#             x_test.append(x_array[n])
#         elif counter >= 8:
#             x_train.append(x_array[n])
#
#         if counter < 5:
#             y_test.append(y_array[n])
#         else:
#             y_train.append(y_array[n])
#
#         if counter == 9:
#             counter = 0
#         else:
#             counter += 1
#
#     return x_train, x_test, y_train, y_test


def data_read_splitting(file):
    data = pd.read_csv(file)

    y_arr = data.iloc[:, 0].values  # Target vector
    x_arr = data.iloc[:, 1:].values  # Features vectors

    X_train, X_test, y_train, y_test = division_dataset(x_arr, y_arr)

    # # Split data into 50% train and 50% test subsets part 2!!
    # stratSplit = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=33)
    # for train_idx, test_idx in stratSplit.split(x_arr, y_arr):
    #     X_train, X_test = x_arr[train_idx], x_arr[test_idx]
    #     y_train, y_test = y_arr[train_idx], y_arr[test_idx]

    return X_train, X_test, y_train, y_test


def main():

    lst_files = ['CASIA_gabor24_24lbp8_8lbp1.csv']

    figure = 1
    for csv_file in lst_files:
        print("\nBeginning of process with " + csv_file)
        X_train, X_test, y_train, y_test = data_read_splitting(csv_file)

        print("\nBeginning of Feature selection")
        param_kbest = SelectKBest(f_classif, k=1000)
        param_kbest.fit(X_train, y_train)
        x_train_kbest = param_kbest.transform(X_train)  # Then we transform both the training an the test set
        x_test_kbest = param_kbest.transform(X_test)

        # Feature normalization with a standard scaler is very useful
        scaler = StandardScaler()
        # The function fit_transform makes both fitting and transformation. It is equivalent to the function fit
        # followed by the function transform
        x_train_kbest = scaler.fit_transform(x_train_kbest)  # Again, the fitting applied only on the training set
        x_test_kbest = scaler.transform(x_test_kbest)  # The test set instead is only transformed

        # ******RANDOM FOREST WITH PARAMETER OPTIMIZATION******
        print("\nBeginning of Random Forest")
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=33)

        rfc = RandomForestClassifier(random_state=42)

        param_grid_rf = {
            'n_estimators': [50, 100, 150],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': [4, 5, 6, 7, 8],
            'criterion': ['gini', 'entropy']
        }
        GS_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid_rf, cv=cv)
        GS_rfc.fit(x_train_kbest, y_train)
        params_best = GS_rfc.best_params_
        print('Best parameters for Random forest for ' + csv_file + ' are = ', params_best)

        y_true, y_pred = y_test, GS_rfc.predict(x_test_kbest)
        y_pred_proba = GS_rfc.predict_proba(x_test_kbest)
        DI_rf = DI_calculation(y_test, y_pred, y_pred_proba)
        print('Decidability Index for Random forest for ' + csv_file + ' file is = ' + str(DI_rf))
        print("Accuracy Random forest for " + csv_file + " file is = ", metrics.accuracy_score(y_test, y_pred))

        title = 'ROC Curves for Random Forest'
        plot_roc_curve(GS_rfc, x_train_kbest, y_train, x_test_kbest, y_test, title, csv_file, figure)

        stats_a = get_eer_stats(y_test, y_pred)
        generate_eer_report([stats_a], [csv_file], 'Final_DD_RF_pyeer_' + csv_file)
        # export_error_rates(stats_a.fmr, stats_a.fnmr, 'A_DET_G24_248_81.csv')

        # ********SVM: Support Vector Classification*******
        print("\nBeginning of SVC")
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=33)

        svc = SVC(random_state=42, probability=True)

        param_grid_svc = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                           'C': [0.01, 1, 10, 100, 1000]},
                          {'kernel': ['linear'], 'C': [0.01, 10, 100, 1000]},
                          {'kernel': ['poly'], 'C': [0.01, 10, 100, 1000]},
                          {'kernel': ['sigmoid'], 'C': [0.01, 10, 100, 1000]}
                          ]

        grid_search_svc = GridSearchCV(estimator=svc, param_grid=param_grid_svc, cv=cv, refit=True)
        grid_search_svc.fit(x_train_kbest, y_train)
        params_best = grid_search_svc.best_params_
        print('Best parameters for SVC for ' + csv_file + ' are = ', params_best)

        y_pred_svc = grid_search_svc.predict(x_test_kbest)
        y_pred_proba = grid_search_svc.predict_proba(x_test_kbest)
        DI_svc = DI_calculation(y_test, y_pred_svc, y_pred_proba)
        print('Decidability Index for SVC for ' + csv_file + ' file is = ' + str(DI_svc))
        print("Accuracy SVC for " + csv_file + " is = ", metrics.accuracy_score(y_test, y_pred_svc))

        title = 'ROC Curves for SVC'
        plot_roc_curve(grid_search_svc, x_train_kbest, y_train, x_test_kbest, y_test, title, csv_file, figure)

        stats_a = get_eer_stats(y_test, y_pred_svc)
        generate_eer_report([stats_a], [csv_file], 'Final_DD_SVC_pyeer_' + csv_file)
        # export_error_rates(stats_a.fmr, stats_a.fnmr, 'A_DET_G24_248_81.csv')

        # **********KNN************
        print("\nBeginning of KNN")
        param_grid_knn = {'n_neighbors': [3, 7, 15, 33]}
        grid_search_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=cv, refit=True)
        grid_search_knn.fit(x_train_kbest, y_train)
        params_best = grid_search_knn.best_params_
        print('Best parameters for SVC for ' + csv_file + ' are = ', params_best)

        y_pred_knn = grid_search_knn.predict(x_test_kbest)
        y_pred_proba = grid_search_knn.predict_proba(x_test_kbest)
        DI_knn = DI_calculation(y_test, y_pred_knn, y_pred_proba)
        print('Decidability Index for kNN for ' + csv_file + ' file is = ' + str(DI_knn))
        print("Accuracy kNN for " + csv_file + " is = ", metrics.accuracy_score(y_test, y_pred_knn))

        title = 'ROC Curves for kNN'
        plot_roc_curve(grid_search_knn, x_train_kbest, y_train, x_test_kbest, y_test, title, csv_file, figure)

        stats_a = get_eer_stats(y_test, y_pred_knn)
        generate_eer_report([stats_a], [csv_file], 'Final_DD_kNN_pyeer_' + csv_file)
        figure += 1
        # export_error_rates(stats_a.fmr, stats_a.fnmr, 'A_DET_G24_248_81.csv')


if __name__ == '__main__':
    start = timer()
    main()
    print("\nTime taken by the algorithm:", timer() - start)
