from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class Fading(Hook):
    def __init__(self, fade_epoch = 100000):
        self.fade_epoch = fade_epoch

    def before_train_epoch(self, runner):
        if runner.epoch >= self.fade_epoch:
            if not hasattr(runner.data_loader.dataset, 'pipeline'):
                for i, transform in enumerate(runner.data_loader.dataset.dataset.pipeline.transforms):
                    if type(transform).__name__ == 'UnifiedObjectSample':
                        runner.data_loader.dataset.dataset.pipeline.transforms.pop(i)
                        break
            else:
                for i, transform in enumerate(runner.data_loader.dataset.pipeline.transforms):
                    if type(transform).__name__ == 'UnifiedObjectSample':
                        runner.data_loader.dataset.pipeline.transforms.pop(i)
                        break
                if hasattr(runner.data_loader.dataset, 'det_pipeline') and runner.data_loader.dataset.det_pipeline is not None:
                    for i, transform in enumerate(runner.data_loader.dataset.det_pipeline.transforms):
                        if type(transform).__name__ == 'UnifiedObjectSample':
                            runner.data_loader.dataset.det_pipeline.transforms.pop(i)
                            break