from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class LowLevelModelsHandler():
    """"""

    def __init__(self):
        self.mapper = self._generate_models_mapper()

    def _generate_models_mapper(self):
        mapper = {
            'random_forest': [RandomForestClassifier, RandomForestRegressor],
            'svm': [SVC, SVR],
            'knn': [KNeighborsClassifier, KNeighborsRegressor],
        }
        return mapper

    def get_model(self, model, parameters, type_='classification'):
        index = False
        if type_ == 'classification':
            index = 0
        elif type_ == 'regression':
            index = 1

        low_level_model = self.mapper[model][index]
        low_level_model.set_params(parameters)

        return low_level_model()
