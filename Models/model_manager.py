import pickle

class ModelManager:
    def save_model(self, modelName, Q):
        filepath = f'../Models/{modelName}.pkl'
        with open(filepath, 'wb') as file:
            pickle.dump(Q, file)

    def load_model(self, modelName):
        filepath = f'../Models/{modelName}.pkl'
        with open(filepath, 'rb') as file:
            file = pickle.load(file)
            return file