from hyperopt import hp


d = {
        'p1': hp.choice('p1_op', [True, False]),
        'p2': hp.choice('p2_op', ['avg', 'sum'])
    }

# data_choices = hp.choice('data', d)
data_choices = None

prep_choices = {
    'type': hp.choice('preprocessing',['standard','normalize','pca'])
}
