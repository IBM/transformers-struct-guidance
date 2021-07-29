class EarlyStopping:

    def __init__(self, best_validation_loss, no_improvement_count, threshold=2):
        self.best_validation_loss = best_validation_loss
        self.counter = no_improvement_count
        self.threshold = threshold

    def check_stopping_criterion(self, validation_loss):
        if validation_loss >= self.best_validation_loss:
            self.counter += 1
        else:
            self.best_validation_loss = validation_loss
            self.counter = 0

        if self.counter >= self.threshold:
            return True
        else:
            return False
