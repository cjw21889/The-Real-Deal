import os
import numpy as np
import pandas as pd
# sklearn
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, FunctionTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor, StackingRegressor, VotingRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.svm import SVR, LinearSVR, NuSVR
# Other ML packages
from xgboost import XGBRegressor
# bespoke pacakge
from trd.data import load_data
from trd.encoder import RatingTransformer, ProductionTransformer, AgeTransformer, CountryTransformer, RunTimeTransformer, LanguageTransformer, DateTransformer
from trd.params import MLFLOW_URI
# MLflow
from memoized_property import memoized_property
from mlflow.tracking import MlflowClient
import mlflow
# misc
from termcolor import colored
import joblib



EXPERIMENT_NAME = "Pris_LND_Netflix_experiment"


class Trainer(object):
    ESTIMATOR = "Linear"
    def __init__(self, X, y, **kwargs):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        #self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.3)
        self.X = X
        self.y = y
        self.kwargs = kwargs
        # for MLFlow
        self.experiment_name = EXPERIMENT_NAME

    def set_experiment_name(self, experiment_name):
        '''defines the experiment name for MLFlow'''
        self.experiment_name = experiment_name
    def get_estimator(self):
        estimator = self.kwargs.get("estimator", self.ESTIMATOR)
        # set a base for bagging models
        base = self.kwargs.get('basemodel', 'Tree')
        if base == 'Tree':
            base_model = DecisionTreeRegressor()
        else:
            base_model = LinearRegression()

        if estimator == "Lasso":
            model = Lasso()
        elif estimator == "Ridge":
            model = Ridge()
        elif estimator == "Linear":
            model = LinearRegression()
        elif estimator == "Neighbors":
            model = KNeighborsRegressor()
            self.model_params = {'n_neighbors': list(range(5, 50, 5))}
        elif estimator == "Tree":
            model = DecisionTreeRegressor()
            self.model_params = {'max_features': ['auto', 'sqrt', 0.2, 0.3, 0.4, 0.5],
                                 'max_depth': list(range(7, 71, 10)),
                                 'min_samples_leaf': list(range(3, 51, 7))}
        elif estimator == "RandomForest":
            model = RandomForestRegressor()
            self.model_params = {'max_features': ['auto', 'sqrt', 0.2, 0.3, 0.4, 0.5],
                                'max_depth': list(range(7,71,10)),
                                'min_samples_leaf': list(range(3,51,7)),
                                'n_estimators': list(range(100, 451, 50))}
        elif estimator == 'LightGBR':
            model = HistGradientBoostingRegressor()
        elif estimator == "GBM":
            model = GradientBoostingRegressor()
            self.model_params = {'max_features': ['auto', 'sqrt', 0.2, 0.3, 0.4, 0.5],
                                 'max_depth': list(range(7, 71, 10)),
                                 'min_samples_leaf': list(range(3, 51, 7)),
                                 'n_estimators': list(range(100, 451, 50))}
        elif estimator == "SVR":
            model = SVR()
            self.model_params = {'kernel': ['rbf'],
                                 'gamma': np.linspace(0.0001, 10, 10),
                                 'C': np.linspace(0.1, 20, 10)}
        elif estimator == "Xgboost":
            model = XGBRegressor(objective='reg:squarederror', booster='gbtree',n_jobs=-1)
            self.model_params = {'max_depth': range(2, 20, 2),
                                 'n_estimators': range(60, 220, 40),
                                 'learning_rate': [0.3, 0.1, 0.01, 0.05],
                                 'min_child_weight': [1, 3, 5],
                                 'gamma': [1, 3, 5]}
        elif estimator == "Bagging":
            model = BaggingRegressor(base_model, n_estimators=50)
        elif estimator == "Stacking":
            model = StackingRegressor(estimators=[('neighbors', KNeighborsRegressor()),
                                                  ('svr', SVR()), ('tree', DecisionTreeRegressor(min_samples_split=25, min_samples_leaf=40, max_depth=50))])
        elif estimator == "Voting":
            model = VotingRegressor(estimators=[('neighbors', KNeighborsRegressor()),
                                                ('svr', SVR()), ('tree', DecisionTreeRegressor(min_samples_split=25, min_samples_leaf=40, max_depth=50))])
        elif estimator == "ADA":
            model = AdaBoostRegressor(base_model)
        else:
            model = Lasso()
        estimator_params = self.kwargs.get("estimator_params", {})
        self.mlflow_log_param("estimator", estimator)
        model.set_params(**estimator_params)
        print(colored(model.__class__.__name__, "red"))
        return model




    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        full_name_token = '[a-zA-Z][a-z -]+'
        one_dim = FunctionTransformer(np.reshape, kw_args={'newshape': -1})
        feateng_steps = self.kwargs.get('feat_eng', ['Actors, Runtime'])
        # Count vectorizers:
        pipe_actor = Pipeline([
            ('actor_imputer', SimpleImputer(fill_value='Unknown', strategy='constant')),
            ('actor_reshape', one_dim),
            ('actor_count', CountVectorizer(token_pattern=full_name_token))
            ])

        pipe_genre = Pipeline([
            ('genre_imputer', SimpleImputer(fill_value='Unknown', strategy='constant')),
            ('genre_reshape', one_dim),
            ('genre_count', CountVectorizer(token_pattern=full_name_token))
            ])

        pipe_writer = Pipeline([
            ('wrtier_imputer', SimpleImputer(fill_value='Unknown', strategy='constant')),
            ('writer_reshape', one_dim),
            ('writer_count', CountVectorizer(token_pattern=full_name_token))
            ])

        pipe_director = Pipeline([
            ('director_imputer', SimpleImputer(fill_value='Unknown', strategy='constant')),
            ('director_reshape', one_dim),
            ('director_count', CountVectorizer(token_pattern=full_name_token))
            ])
        # production company
        pipe_production = Pipeline([
            ('production_imputer', SimpleImputer(fill_value='Unknown', strategy='constant')),
            ('production_transformer', ProductionTransformer()),
            ('production_reshape', one_dim),
            ('production_count', CountVectorizer(
                token_pattern=full_name_token))
            ])
        # Rating maping and encoding
        pipe_rated = Pipeline([
            ('rated_imputer', SimpleImputer(fill_value='Not Rated', strategy='constant')),
            ('rated_map', RatingTransformer()),
            ('rated_OHE', OneHotEncoder(handle_unknown='ignore', sparse=False))
            ])
        # runtime clean and scale
        pipe_runtime = Pipeline([
            ('rated_imputer', SimpleImputer(fill_value='90', strategy='constant')),
            ('runtime_clean', RunTimeTransformer()),
            ('runtime_scale', MinMaxScaler())
            ])
        # language binary for english
        pipe_language = Pipeline([
            ('language_imputer', SimpleImputer(fill_value='Unknown', strategy='constant')),
            ('language_transform', LanguageTransformer())
            ])
        # released month encoded
        pipe_released = Pipeline([
            ('released_imputer', SimpleImputer(fill_value='Unknown', strategy='constant')),
            ('released_transform', DateTransformer()),
            ('released_OHE', OneHotEncoder(handle_unknown='ignore', sparse=False))
            ])
        # age
        pipe_age = Pipeline([
            ('age_imputer', SimpleImputer(
                fill_value=2021, strategy='constant')),
            ('age_transform', AgeTransformer()),
            ('age_scaler', MinMaxScaler())
        ])
        # country binary for us
        pipe_country = Pipeline([
            ('country_imputer', SimpleImputer(
                fill_value='Unknown', strategy='constant')),
            ('country_transform', CountryTransformer()),
            ])

        # pipe_plot = Pipeline([(     )])

        feateng_blocks = [('actor', pipe_actor, ['Actors']),
                        ('director', pipe_director,['Director']),
                        ('writer', pipe_writer, ['Writer']),
                        ('genre', pipe_genre, ['Genre']),
                        ('language', pipe_language, ['Language']),
                        # ('country', pipe_country, ['Country']),
                        # ('released', pipe_released, ['Released']),
                        ('age', pipe_age, ['Year']),
                        ('rated', pipe_rated,['Rated']),
                        ('production', pipe_production, ['Production']),
                        ('runtime', pipe_runtime, ['Runtime'])
                        ]
        # Filter out some bocks according to input parameters
        for bloc in feateng_blocks:
            if bloc[0] not in feateng_steps:
                feateng_blocks.remove(bloc)

        # create column transformer with just bloc pieces you want
        feat_encoder = ColumnTransformer(feateng_blocks, n_jobs=-1, remainder='drop')

        # set a final pipeline
        self.pipeline = Pipeline([
            ('preprocessor', feat_encoder),
            ('model', self.get_estimator())
        ])

    def add_grid_search(self):
        """"
        Apply Gridsearch on self.params defined in get_estimator
        {'rgs__n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
          'rgs__max_features' : ['auto', 'sqrt'],
          'rgs__max_depth' : [int(x) for x in np.linspace(10, 110, num = 11)]}
        """
        # Here to apply ramdom search to pipeline, need to follow naming "rgs__paramname"
        params = {"model__" + k: v for k, v in self.model_params.items()}
        for sub_pipe in ['actor', 'director', 'writer', 'production', 'genre']:
            params[f"preprocessor__{sub_pipe}__{sub_pipe}_count__max_features"] = np.arange(50)
        self.pipeline = RandomizedSearchCV(estimator=self.pipeline, param_distributions=params,
                                           n_iter=75,
                                           cv=5,
                                           verbose=10,
                                           random_state=41,
                                           n_jobs=-1)
    def run(self):
        self.set_pipeline()
        self.gridsearch = self.kwargs.get('gridsearch', False)
        if self.gridsearch:
            self.add_grid_search()
        self.pipeline.fit(self.X, self.y)
        if self.gridsearch:
            for k, v in self.pipeline.best_params_.items():
                self.mlflow_log_param(k, v)
        print('pipeline fitted')


    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on X and return the RMSE"""
        if self.kwargs.get('permutation', False):
            permutation_score = permutation_importance(
                self.pipeline, self.X, self.y, n_repeats=10)  # Perform Permutation
            importance_df = pd.DataFrame(np.vstack((self.X.columns,
                                        permutation_score.importances_mean)).T)  # Unstack results
            importance_df.columns = ['feature', 'score decrease']

            print(importance_df.sort_values(by="score decrease", ascending=False))
        y_pred_train = self.pipeline.predict(self.X)
        print(y_pred_train[y_pred_train.argmax()])
        mse_train = mean_squared_error(self.y, y_pred_train)
        rmse_train = np.sqrt(mse_train)

        self.mlflow_log_metric("rmse_train", rmse_train)

        y_pred_test = self.pipeline.predict(X_test)
        mse_test = mean_squared_error(y_test, y_pred_test)
        rmse_test = np.sqrt(mse_test)
        self.mlflow_log_metric("rmse_test", rmse_test)

        return (round(rmse_train, 3), round(rmse_test, 3))

    def predict(self, X):
        y_pred = self.pipeline.predict(X)
        return y_pred

    def save_model(self):
        """Save the model into a .joblib format"""
        joblib.dump(self.pipeline, 'model.joblib')
        print(colored("model.joblib saved locally", "green"))

 # MLFlow methods
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(
                self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == "__main__":
    N = 10000
    df = load_data(N, movies_only=True)
    print(df.avg_review_score.max())
    print(df.sort_values(by='avg_review_score').tail(2))
    print(df.avg_review_score.argmax())
    print(df.iloc[df.avg_review_score.argmax()])
    drops = ['avg_review_score', 'n_reviews', 'year', 'title', 'Title', 'totalSeasons', 'Response', 'Ratings',
             'Awards', 'Poster', 'Metascore', 'imdbRating', 'imdbVotes', 'imdbID', 'Type', 'Metacritic',
             'Internet Movie Database', 'Index_match', 'Released', 'Country', 'DVD', 'BoxOffice', 'Website', 'Rotten Tomatoes']
    X = df.drop(columns=drops)
    y = df.avg_review_score
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    #Train and save model, locally and
    models = ["Lasso", 'Ridge', 'Linear', 'Neighbors', 'Tree', 'Bagging', 'Stacking',
              'Voting', 'ADA', 'GBM', 'LightGBR','SVR', 'RandomForest', 'Xgboost']
    single_models = ['Neighbors', 'Tree', 'RanomForest', 'GBM', 'Xgboost', 'SVR']
    all_features = ['Year', 'Rated', 'Runtime', 'Genre', 'Director', 'Writer', 'Actors',
                    'Plot', 'Language', 'Production']
    run_features = ['age', 'rated', 'released', 'runtime', 'genre', 'director', 'writer', 'actors'
                    'plot', 'language', 'country','production']
    for mod in single_models:
        pass

    params = {
        'estimator': 'Xgboost',
        'feat_eng': run_features,
        'permutation': True,
        'gridsearch': False
        }
    max = y_train.argmax()
    print(X_train.loc[max])
    print(y_train.loc[max])
    trainer = Trainer(X_train, y_train, **params)
    trainer.set_experiment_name('Single_model_loop_6.13')
    trainer.run()
    rmse = trainer.evaluate(X_test, y_test)
    print(f"rmse: {rmse}")
    trainer.save_model()
