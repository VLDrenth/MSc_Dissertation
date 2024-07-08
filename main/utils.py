import pickle

def save_experiment(file_name, parameters, outcome):
    file_path = f'./experiments/{file_name}.pkl'
    data = {
        'parameters': parameters,
        'outcome': outcome
    }
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_experiment(file_name):
    file_path = f'./experiments/{file_name}.pkl'
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data['parameters'], data['outcome']


