import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor


# functions
def check_min_error(current_mae, min_mae, verbiage):
    if min_mae is None:
        min_mae = {'value': (current_mae + 1), 'verbiage': ""}

    if current_mae < min_mae['value']:
        print("New MAE found")
        print(current_mae)
        print(verbiage)

        min_mae = {'value': current_mae, 'verbiage': verbiage}
    return min_mae

def treat_data_dropna(train_X, validation_X):
    cols_with_missing = [col for col in train_X.columns if train_X[col].isnull().any()]
    reduced_train_X = train_X.drop(cols_with_missing, axis=1)
    reduced_validation_X = validation_X.drop(cols_with_missing, axis=1)
    return reduced_train_X, reduced_validation_X

def treat_data_impute_zero(train_X, validation_X):
    my_imputer = SimpleImputer(strategy='constant', fill_value=0.0)
    imputed_X_train = pd.DataFrame(my_imputer.fit_transform(train_X))
    imputed_X_validation = pd.DataFrame(my_imputer.transform(validation_X))
    imputed_X_train.columns = train_X.columns
    imputed_X_validation.columns = validation_X.columns
    return imputed_X_train, imputed_X_validation

def treat_data_impute_mean(train_X, validation_X):
    my_imputer = SimpleImputer(strategy='mean')
    imputed_X_train = pd.DataFrame(my_imputer.fit_transform(train_X))
    imputed_X_validation = pd.DataFrame(my_imputer.transform(validation_X))
    imputed_X_train.columns = train_X.columns
    imputed_X_validation.columns = validation_X.columns
    return imputed_X_train, imputed_X_validation

def treat_data_impute_median(train_X, validation_X):
    my_imputer = SimpleImputer(strategy='median')
    imputed_X_train = pd.DataFrame(my_imputer.fit_transform(train_X))
    imputed_X_validation = pd.DataFrame(my_imputer.transform(validation_X))
    imputed_X_train.columns = train_X.columns
    imputed_X_validation.columns = validation_X.columns
    return imputed_X_train, imputed_X_validation

def treat_data_ohe(train_X, validation_X):
    ohe_train_X = pd.get_dummies(train_X)
    ohe_validation_X = pd.get_dummies(validation_X)

    # using inner vs left has a huge impact on model performance
    # with left, the score for this case was worse
    ohe_aligned_train_X, ohe_aligned_validation_X = ohe_train_X.align(ohe_validation_X, join='inner', axis=1)

    return ohe_aligned_train_X, ohe_aligned_validation_X

def treat_data_dropcat(train_X, validation_X):
    stripped_train_X = train_X.select_dtypes(exclude=['object'])
    stripped_validation_X = validation_X.select_dtypes(exclude=['object'])
    return stripped_train_X, stripped_validation_X

# # main loop
# load data
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

#generating data sets
data = train.copy()

vectors = [
    'NU_IDADE',
    'TP_SEXO',
    'TP_COR_RACA',
    'TP_NACIONALIDADE',
    'TP_ST_CONCLUSAO',
    'TP_ANO_CONCLUIU',
    'TP_ESCOLA',
    'TP_ENSINO',
    'IN_TREINEIRO',
    'TP_DEPENDENCIA_ADM_ESC',
    'IN_BAIXA_VISAO',
    'IN_CEGUEIRA',
    'IN_SURDEZ',
    'IN_DISLEXIA',
    'IN_DISCALCULIA',
    'IN_SABATISTA',
    'IN_GESTANTE',
    'IN_IDOSO',
    'TP_PRESENCA_CN',
    'TP_PRESENCA_CH',
    'TP_PRESENCA_LC',
    'NU_NOTA_CN',
    'NU_NOTA_CH',
    'NU_NOTA_LC',
    'TP_LINGUA',
    'TP_STATUS_REDACAO',
    'NU_NOTA_COMP1',
    'NU_NOTA_COMP2',
    'NU_NOTA_COMP3',
    'NU_NOTA_COMP4',
    'NU_NOTA_COMP5',
    'NU_NOTA_REDACAO',
    'Q001',
    'Q002',
    'Q006',
    'Q024',
    'Q025',
    'Q026',
    'Q027',
    'Q047'
]

# y = data.NU_NOTA_MT
# y.fillna(0.0, inplace=True)
#
# X = data.loc[:, vectors]
#
# min_score = None
#
# train_X, validation_X, train_y, validation_y = train_test_split(X, y, random_state = 0)
#
# # Set 1 - dropping NAs, dropping categorical
# train_partial, validation_partial = treat_data_dropna(train_X, validation_X)
# train_dropped_dropped, validation_dropped_dropped = treat_data_dropcat(train_partial, validation_partial)
#
# # Set 2 - dropping NAs, OHE categorical
# train_partial, validation_partial = treat_data_dropna(train_X, validation_X)
# train_dropped_encoded, validation_dropped_encoded = treat_data_ohe(train_partial, validation_partial)
#
# # Set 3 - using Imputer Zero for NAs, dropping categorical
# train_partial, validation_partial = treat_data_dropcat(train_X, validation_X)
# train_impute_0_dropped, validation_impute_0_dropped = treat_data_impute_zero(train_partial, validation_partial)
#
# # Set 4 - using Imputer Zero for NAs, OHE categorical
# train_partial, validation_partial = treat_data_ohe(train_X, validation_X)
# train_impute_0_encoded, validation_impute_0_encoded = treat_data_impute_zero(train_partial, validation_partial)
#
# # Set 5 - using Imputer Mean for NAs, dropping categorical
# train_partial, validation_partial = treat_data_dropcat(train_X, validation_X)
# train_impute_mean_dropped, validation_impute_mean_dropped = treat_data_impute_mean(train_partial, validation_partial)
#
# # Set 6 - using Imputer Mean for NAs, OHE categorical
# train_partial, validation_partial = treat_data_ohe(train_X, validation_X)
# train_impute_mean_encoded, validation_impute_mean_encoded = treat_data_impute_mean(train_partial, validation_partial)
#
# # Set 7 - using Imputer Mean for NAs, dropping categorical
# train_partial, validation_partial = treat_data_dropcat(train_X, validation_X)
# train_impute_median_dropped, validation_impute_median_dropped = treat_data_impute_median(train_partial, validation_partial)
#
# # Set 8 - using Imputer Mean for NAs, OHE categorical
# train_partial, validation_partial = treat_data_ohe(train_X, validation_X)
# train_impute_median_encoded, validation_impute_median_encoded = treat_data_impute_median(train_partial, validation_partial)
#
# # create and evaluate models
#
# # RandomForest
# print(' ##### Random Forest ##### ')
# rf_1 = RandomForestRegressor()
# rf_2 = RandomForestRegressor()
# rf_3 = RandomForestRegressor()
# rf_4 = RandomForestRegressor()
# rf_5 = RandomForestRegressor()
# rf_6 = RandomForestRegressor()
# rf_7 = RandomForestRegressor()
# rf_8 = RandomForestRegressor()
#
# print('Data dropped missing, dropped categorical')
# rf_1.fit(train_dropped_dropped, train_y)
# rf_1_predicts = rf_1.predict(validation_dropped_dropped)
# current_mae = mean_absolute_error(rf_1_predicts, validation_y)
# print(current_mae)
# verbiage = "Random Forest with data => dropped missings, dropped categorical"
# min_score = check_min_error(current_mae, min_score, verbiage)
#
# print('Data dropped missing, encoded categorical')
# rf_2.fit(train_dropped_encoded, train_y)
# rf_2_predicts = rf_2.predict(validation_dropped_encoded)
# current_mae = mean_absolute_error(rf_2_predicts, validation_y)
# print(current_mae)
# verbiage = "Random Forest with data => dropped missings, encoded categorical"
# min_score = check_min_error(current_mae, min_score, verbiage)
#
# print('Data imputed missing (0), dropped categorical')
# rf_3.fit(train_impute_0_dropped, train_y)
# rf_3_predicts = rf_3.predict(validation_impute_0_dropped)
# current_mae = mean_absolute_error(rf_3_predicts, validation_y)
# print(current_mae)
# verbiage = "Random Forest with data => imputed missing (0), dropped categorical"
# min_score = check_min_error(current_mae, min_score, verbiage)
#
# print('Data imputed missing (0), encoded categorical')
# rf_4.fit(train_impute_0_encoded, train_y)
# rf_4_predicts = rf_4.predict(validation_impute_0_encoded)
# current_mae = mean_absolute_error(rf_4_predicts, validation_y)
# print(current_mae)
# verbiage = "Random Forest with data => imputed missing (0), encoded categorical"
# min_score = check_min_error(current_mae, min_score, verbiage)
#
# print('Data imputed missing (mean), dropped categorical')
# rf_5.fit(train_impute_mean_dropped, train_y)
# rf_5_predicts = rf_5.predict(validation_impute_mean_dropped)
# current_mae = mean_absolute_error(rf_5_predicts, validation_y)
# print(current_mae)
# verbiage = "Random Forest with data => imputed missing (mean), dropped categorical"
# min_score = check_min_error(current_mae, min_score, verbiage)
#
# print('Data imputed missing (mean), encoded categorical')
# rf_6.fit(train_impute_mean_encoded, train_y)
# rf_6_predicts = rf_6.predict(validation_impute_mean_encoded)
# current_mae = mean_absolute_error(rf_6_predicts, validation_y)
# print(current_mae)
# verbiage = "Random Forest with data => imputed missing (mean), encoded categorical"
# min_score = check_min_error(current_mae, min_score, verbiage)
#
# print('Data imputed missing (median), dropped categorical')
# rf_7.fit(train_impute_median_dropped, train_y)
# rf_7_predicts = rf_7.predict(validation_impute_median_dropped)
# current_mae = mean_absolute_error(rf_7_predicts, validation_y)
# print(current_mae)
# verbiage = "Random Forest with data => imputed missing (median), dropped categorical"
# min_score = check_min_error(current_mae, min_score, verbiage)
#
# print('Data imputed missing (median), encoded categorical')
# rf_8.fit(train_impute_median_encoded, train_y)
# rf_8_predicts = rf_8.predict(validation_impute_median_encoded)
# current_mae = mean_absolute_error(rf_8_predicts, validation_y)
# print(current_mae)
# verbiage = "Random Forest with data => imputed missing (median), encoded categorical"
# min_score = check_min_error(current_mae, min_score, verbiage)
#
#
# # Decision Trees
# print(' ##### Decision Trees ##### ')
#
# leaf_nodes_amount = [64, 128, 256, 512, 1024, 2048, 4096]
# for x in leaf_nodes_amount:
#     print("\n## Using {} Leaf Nodes ## ".format(x))
#     rf_1 = DecisionTreeRegressor(max_leaf_nodes=x, random_state=0)
#     rf_2 = DecisionTreeRegressor(max_leaf_nodes=x, random_state=0)
#     rf_3 = DecisionTreeRegressor(max_leaf_nodes=x, random_state=0)
#     rf_4 = DecisionTreeRegressor(max_leaf_nodes=x, random_state=0)
#     rf_5 = DecisionTreeRegressor(max_leaf_nodes=x, random_state=0)
#     rf_6 = DecisionTreeRegressor(max_leaf_nodes=x, random_state=0)
#     rf_7 = DecisionTreeRegressor(max_leaf_nodes=x, random_state=0)
#     rf_8 = DecisionTreeRegressor(max_leaf_nodes=x, random_state=0)
#
#     print('Data dropped missing, dropped categorical')
#     rf_1.fit(train_dropped_dropped, train_y)
#     rf_1_predicts = rf_1.predict(validation_dropped_dropped)
#     current_mae = mean_absolute_error(rf_1_predicts, validation_y)
#     print(current_mae)
#     verbiage = "Decision Tree - {} nodes, with data => dropped missing, dropped categorical".format(x)
#     min_score = check_min_error(current_mae, min_score, verbiage)
#
#     print('Data dropped missing, encoded categorical')
#     rf_2.fit(train_dropped_encoded, train_y)
#     rf_2_predicts = rf_2.predict(validation_dropped_encoded)
#     current_mae = mean_absolute_error(rf_2_predicts, validation_y)
#     print(current_mae)
#     verbiage = "Decision Tree - {} nodes, with data => dropped missing, encoded categorical".format(x)
#     min_score = check_min_error(current_mae, min_score, verbiage)
#
#     print('Data imputed missing (0), dropped categorical')
#     rf_3.fit(train_impute_0_dropped, train_y)
#     rf_3_predicts = rf_3.predict(validation_impute_0_dropped)
#     current_mae = mean_absolute_error(rf_3_predicts, validation_y)
#     print(current_mae)
#     verbiage = "Decision Tree - {} nodes, with data => imputed missing (0), dropped categorical".format(x)
#     min_score = check_min_error(current_mae, min_score, verbiage)
#
#     print('Data imputed missing (0), encoded categorical')
#     rf_4.fit(train_impute_0_encoded, train_y)
#     rf_4_predicts = rf_4.predict(validation_impute_0_encoded)
#     current_mae = mean_absolute_error(rf_4_predicts, validation_y)
#     print(current_mae)
#     verbiage = "Decision Tree - {} nodes, with data => imputed missing (0), encoded categorical".format(x)
#     min_score = check_min_error(current_mae, min_score, verbiage)
#
#     print('Data imputed missing (mean), dropped categorical')
#     rf_5.fit(train_impute_mean_dropped, train_y)
#     rf_5_predicts = rf_5.predict(validation_impute_mean_dropped)
#     current_mae = mean_absolute_error(rf_5_predicts, validation_y)
#     print(current_mae)
#     verbiage = "Decision Tree - {} nodes, with data => imputed missing (mean), dropped categorical".format(x)
#     min_score = check_min_error(current_mae, min_score, verbiage)
#
#     print('Data imputed missing (mean), encoded categorical')
#     rf_6.fit(train_impute_mean_encoded, train_y)
#     rf_6_predicts = rf_6.predict(validation_impute_mean_encoded)
#     current_mae = mean_absolute_error(rf_6_predicts, validation_y)
#     print(current_mae)
#     verbiage = "Decision Tree - {} nodes, with data => imputed missing (mean), encoded categorical".format(x)
#     min_score = check_min_error(current_mae, min_score, verbiage)
#
#     print('Data imputed missing (median), dropped categorical')
#     rf_7.fit(train_impute_median_dropped, train_y)
#     rf_7_predicts = rf_7.predict(validation_impute_median_dropped)
#     current_mae = mean_absolute_error(rf_7_predicts, validation_y)
#     print(current_mae)
#     verbiage = "Decision Tree - {} nodes, with data => imputed missing (median), dropped categorical".format(x)
#     min_score = check_min_error(current_mae, min_score, verbiage)
#
#     print('Data imputed missing (median), encoded categorical')
#     rf_8.fit(train_impute_median_encoded, train_y)
#     rf_8_predicts = rf_8.predict(validation_impute_median_encoded)
#     current_mae = mean_absolute_error(rf_8_predicts, validation_y)
#     print(current_mae)
#     verbiage = "Decision Tree - {} nodes, with data => imputed missing (median), encoded categorical".format(x)
#     min_score = check_min_error(current_mae, min_score, verbiage)
#
# # XGBoost
# print(' ##### XGBoost ##### ')
#
# learning_rates = [0.10, 0.15, 0.20, 0.25, 0.30]
# n_estimators = [200, 500, 1000, 2000, 5000]
#
# for i in learning_rates:
#     for j in n_estimators:
#         print("\n## Using {} as learning rate and {} estimators ## ".format(i, j))
#         rf_1 = XGBRegressor(n_estimators=j, learning_rate=i)
#         rf_2 = XGBRegressor(n_estimators=j, learning_rate=i)
#         rf_3 = XGBRegressor(n_estimators=j, learning_rate=i)
#         rf_4 = XGBRegressor(n_estimators=j, learning_rate=i)
#         rf_5 = XGBRegressor(n_estimators=j, learning_rate=i)
#         rf_6 = XGBRegressor(n_estimators=j, learning_rate=i)
#         rf_7 = XGBRegressor(n_estimators=j, learning_rate=i)
#         rf_8 = XGBRegressor(n_estimators=j, learning_rate=i)
#
#         print('Data dropped missing, dropped categorical')
#         rf_1.fit(train_dropped_dropped, train_y, early_stopping_rounds=5, eval_set=[(validation_dropped_dropped, validation_y)], verbose=False)
#         rf_1_predicts = rf_1.predict(validation_dropped_dropped)
#         current_mae = mean_absolute_error(rf_1_predicts, validation_y)
#         print(current_mae)
#         verbiage = "XGBoost - {} learning rate, {} estimators, with data => dropped missing, dropped categorical".format(i, j)
#         min_score = check_min_error(current_mae, min_score, verbiage)
#
#         print('Data dropped missing, encoded categorical')
#         rf_2.fit(train_dropped_encoded, train_y, early_stopping_rounds=5,
#                  eval_set=[(validation_dropped_encoded, validation_y)], verbose=False)
#         rf_2_predicts = rf_2.predict(validation_dropped_encoded)
#         current_mae = mean_absolute_error(rf_2_predicts, validation_y)
#         print(current_mae)
#         verbiage = "XGBoost - {} learning rate, {} estimators, with data => dropped missing, encoded categorical".format(
#             i, j)
#         min_score = check_min_error(current_mae, min_score, verbiage)
#
#         print('Data imputed missing (0), dropped categorical')
#         rf_3.fit(train_impute_0_dropped, train_y, early_stopping_rounds=5,
#                  eval_set=[(validation_impute_0_dropped, validation_y)], verbose=False)
#         rf_3_predicts = rf_3.predict(validation_impute_0_dropped)
#         current_mae = mean_absolute_error(rf_3_predicts, validation_y)
#         print(current_mae)
#         verbiage = "XGBoost - {} learning rate, {} estimators, with data => imputed missing (0), dropped categorical".format(
#             i, j)
#         min_score = check_min_error(current_mae, min_score, verbiage)
#
#         print('Data imputed missing (0), encoded categorical')
#         rf_4.fit(train_impute_0_encoded, train_y, early_stopping_rounds=5,
#                  eval_set=[(validation_impute_0_encoded, validation_y)], verbose=False)
#         rf_4_predicts = rf_4.predict(validation_impute_0_encoded)
#         current_mae = mean_absolute_error(rf_4_predicts, validation_y)
#         print(current_mae)
#         verbiage = "XGBoost - {} learning rate, {} estimators, with data => imputed missing (0), encoded categorical".format(
#             i, j)
#         min_score = check_min_error(current_mae, min_score, verbiage)
#
#         print('Data imputed missing (mean), dropped categorical')
#         rf_5.fit(train_impute_0_dropped, train_y, early_stopping_rounds=5,
#                  eval_set=[(validation_impute_0_dropped, validation_y)], verbose=False)
#         rf_5_predicts = rf_5.predict(validation_impute_0_dropped)
#         current_mae = mean_absolute_error(rf_5_predicts, validation_y)
#         print(current_mae)
#         verbiage = "XGBoost - {} learning rate, {} estimators, with data => imputed missing (mean), dropped categorical".format(
#             i, j)
#         min_score = check_min_error(current_mae, min_score, verbiage)
#
#         print('Data imputed missing (mean), encoded categorical')
#         rf_6.fit(train_impute_mean_encoded, train_y, early_stopping_rounds=5,
#                  eval_set=[(validation_impute_mean_encoded, validation_y)], verbose=False)
#         rf_6_predicts = rf_6.predict(validation_impute_mean_encoded)
#         current_mae = mean_absolute_error(rf_6_predicts, validation_y)
#         print(current_mae)
#         verbiage = "XGBoost - {} learning rate, {} estimators, with data => imputed missing (mean), encoded categorical".format(
#             i, j)
#         min_score = check_min_error(current_mae, min_score, verbiage)
#
#         print('Data imputed missing (median), dropped categorical')
#         rf_7.fit(train_impute_median_dropped, train_y, early_stopping_rounds=5,
#                  eval_set=[(validation_impute_median_dropped, validation_y)], verbose=False)
#         rf_7_predicts = rf_7.predict(validation_impute_median_dropped)
#         current_mae = mean_absolute_error(rf_7_predicts, validation_y)
#         print(current_mae)
#         verbiage = "XGBoost - {} learning rate, {} estimators, with data => imputed missing (median), dropped categorical".format(
#             i, j)
#         min_score = check_min_error(current_mae, min_score, verbiage)
#
#         print('Data imputed missing (median), encoded categorical')
#         rf_8.fit(train_impute_median_encoded, train_y, early_stopping_rounds=5,
#                  eval_set=[(validation_impute_median_encoded, validation_y)], verbose=False)
#         rf_8_predicts = rf_8.predict(validation_impute_median_encoded)
#         current_mae = mean_absolute_error(rf_8_predicts, validation_y)
#         print(current_mae)
#         verbiage = "XGBoost - {} learning rate, {} estimators, with data => imputed missing (median), encoded categorical".format(
#             i, j)
#         min_score = check_min_error(current_mae, min_score, verbiage)
#
# print('Winner was:')
# if min_score is not None:
#     print(min_score['verbiage'])
#     print(min_score['value'])

# generate finished work

# Winner was:
# XGBoost - 0.1 learning rate, 200 estimators, with data => imputed missing (0), encoded categorical
# 43.58767151631526
#
# replicating steps to treat data
train_data = train.copy()
test_data_x = test.copy()

test_data = test_data_x.loc[:, vectors]

y = train_data.NU_NOTA_MT
y.fillna(0, inplace=True)
X = train_data.loc[:, vectors]

train_X, validate_X, train_y, validate_y = train_test_split(X, y, random_state = 0)

train_partial, validation_partial = treat_data_impute_zero(train_X, validate_X)
train_dropped_encoded, validation_dropped_encoded = treat_data_ohe(train_partial, validation_partial)

train_partial, validation_partial = treat_data_impute_zero(train_X, test_data)
train_partial_2, validation_data = treat_data_ohe(train_partial, validation_partial)

# align train_X and validation_X with validation_data
train_dropped_encoded, validation_data = train_dropped_encoded.align(validation_data, join='inner', axis=1)
validation_dropped_encoded, validation_data = validation_dropped_encoded.align(validation_data, join='inner', axis=1)
train_dropped_encoded, validation_dropped_encoded = train_dropped_encoded.align(validation_dropped_encoded, join='inner', axis=1)

# building model
model = XGBRegressor(n_estimators=200, learning_rate=0.1)
model.fit(train_dropped_encoded, train_y, early_stopping_rounds=5, eval_set=[(validation_dropped_encoded, validate_y)], verbose=False)

predicted_notas = model.predict(validation_data)
my_submission = pd.DataFrame({'NU_INSCRICAO': test_data_x.NU_INSCRICAO, 'NU_NOTA_MT': predicted_notas})
my_submission.to_csv('./answer.csv', index=False)
