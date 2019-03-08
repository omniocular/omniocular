from .trainers.diff_trainer import DiffTrainer


class TrainerFactory(object):
    """
    Get the corresponding Trainer class for a particular dataset.
    """
    trainer_map = {
        'VulasDiff': DiffTrainer
    }

    @staticmethod
    def get_trainer(dataset_name, model, embedding, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator=None):

        if dataset_name not in TrainerFactory.trainer_map:
            raise ValueError('{} is not implemented.'.format(dataset_name))

        return TrainerFactory.trainer_map[dataset_name](
            model, embedding, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator
        )
