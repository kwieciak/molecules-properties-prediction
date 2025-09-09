import torch


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, path='best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.best_state = None
        self.path = path

    def check(self, validation_loss, model):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(self.best_state, self.path)
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def load_best(self, model, device=torch.device('cpu')):
        if self.best_state is not None:
            model.load_state_dict(torch.load(self.path, map_location=device))
        return model
