from common.evaluators.diff_token_evaluator import DiffTokenEvaluator
from common.evaluators.paired_token_evaluator import PairedTokenEvaluator


class EvaluatorFactory(object):
    """
    Get the corresponding Evaluator class for a particular dataset.
    """
    evaluator_map = {
        'ApacheDiffToken': DiffTokenEvaluator,
        'SpringDiffToken': DiffTokenEvaluator,
        'VulasDiffToken': DiffTokenEvaluator,
        'VulasPairedToken': PairedTokenEvaluator
    }

    @staticmethod
    def get_evaluator(dataset_cls, model, embedding, data_loader, batch_size, device, keep_results=False):
        if data_loader is None:
            return None

        if dataset_cls.NAME not in EvaluatorFactory.evaluator_map:
            raise ValueError('{} is not implemented.'.format(dataset_cls))

        return EvaluatorFactory.evaluator_map[dataset_cls.NAME](
            dataset_cls, model, embedding, data_loader, batch_size, device, keep_results
        )
