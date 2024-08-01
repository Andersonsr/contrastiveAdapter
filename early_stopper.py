from typing import Literal

_objective_choices = Literal["minimize", "maximize"]
_save_choices = Literal["best", "last"]


class EarlyStopper:
    def __init__(self, patience: int, minimal_improvement: float, objective: _objective_choices = 'minimize',
                 save_option: _save_choices = 'last'):
        self.patience = patience
        self.best_score = None
        self.best_model = None
        self.counter = 0
        self.objective = objective
        self.minimal_improvement = minimal_improvement
        self.last_model = None
        self.save_option = save_option

    def improved(self, score):
        if self.best_score is None:
            self.best_score = score
            return True

        elif self.objective == 'minimize':
            return self.best_score - score > self.minimal_improvement

        else:
            return score - self.best_score > self.minimal_improvement

    def continue_training(self, score, model):
        if self.improved(score):
            self.best_score = score
            self.best_model = model
            self.counter = 0
            return True

        else:
            self.counter += 1
            if self.counter > self.patience:
                self.last_model = model
                return False

            return True

    def model_to_save(self):
        if self.save_option == "best":
            return self.best_model
        else:
            return self.last_model


