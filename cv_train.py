from sklearn.model_selection import RepeatedKFold
import numpy as np

class CrossValidationTraining:
    '''
    from cv_train import CrossValidationTraining
    
    callbacks = [
    keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1),
    keras.callbacks.ModelCheckpoint(filepath='best_model.keras', save_best_only=False)
    ]
    
    input_shape = (None, 224, 224, 1)
    model.build(input_shape)
    
    model_eval = CrossValidationTraining(model, inputs, targets)
    scores, history = model_eval.train_evaluate_model(epochs=10, callbacks=callbacks)
    '''
    def __init__(self, model, inputs, targets):
        self.model = model
        self.inputs = inputs
        self.targets = targets


    def train_evaluate_model(self, epochs=10, callbacks=None):
        fold_no = 1
        
        # define per-fold score containers
        acc_per_fold = []
        loss_per_fold = []
        
        # define evaluation procedure
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        
        for train_ix, test_ix in cv.split(self.inputs, self.targets):
            # create a print statement for the fold
            print(f"Training for fold {fold_no} ...")
            # fit model
            history = self.model.fit(self.inputs[train_ix], self.targets[train_ix], 
                                     epochs=epochs, 
                                     verbose=0,
                                     callbacks=callbacks,
                                    )
            
            scores = self.model.evaluate(self.inputs[test_ix], self.targets[test_ix], verbose=0)
            print(f'Score for fold {fold_no}: {self.model.metrics_names[0]} of {scores[0]}; {self.model.metrics_names[1]} of {scores[1]*100}%')
            acc_per_fold.append(scores[1] * 100)
            loss_per_fold.append(scores[0])
            
            fold_no += 1
        
        # == Provide average scores ==
        print('------------------------------------------------------------------------')
        print('Score per fold')
        for i in range(0, len(acc_per_fold)):
            print('------------------------------------------------------------------------')
            print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
        print('------------------------------------------------------------------------')
        print('Average scores for all folds:')
        print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
        print(f'> Loss: {np.mean(loss_per_fold)}')
        print('------------------------------------------------------------------------')
            
        return scores, history