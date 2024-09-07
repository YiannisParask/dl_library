from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold
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


    def train_evaluate_model(self, method, epochs=10, callbacks=None):
        fold_no = 1
        
        # define per-fold score containers
        acc_per_fold = []
        loss_per_fold = []
        cv_methods = ['KFold', 'StratifiedKFold', 'GroupKFold', 'StratifiedGroupKFold']
        
        # define evaluation procedure
        for method in cv_methods:
            if method == 'KFold':
                cv = KFold(n_splits=5, shuffle=True, random_state=42)
            elif method == 'StratifiedKFold':
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            elif method == 'GroupKFold':
                cv = GroupKFold(n_splits=5)
            elif method == 'StratifiedGroupKFold':
                cv = StratifiedGroupKFold(n_splits=5)
            else:
                raise ValueError('Invalid method. Please select from: KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold')
        
        for train_ix, test_ix in cv.split(self.inputs, self.targets):
            # create a print statement for the fold
            print(f"Training for fold {fold_no} ...")
            # fit model 
            history = self.model.fit(self.inputs[train_ix], self.targets[train_ix], 
                                     epochs=epochs, 
                                     verbose=1,
                                     callbacks=callbacks,
                                     validation_split=0.2
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