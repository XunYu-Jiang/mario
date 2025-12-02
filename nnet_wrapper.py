class NNetWrapper():
    def __init__(self, game, args):
        raise NotImplementedError

    def train(self, examples):
        raise NotImplementedError
    
    def predict(self, obs):
        ...
    
    def save_checkpoint(self, folder, filename):
        ...

    def load_checkpoint(self, folder, filename):
        ...