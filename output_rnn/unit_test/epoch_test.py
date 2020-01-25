''' epoch_test.py '''
from src.train import init_pipeline, load_train_data, load_model, \
                      init_metrics, train_and_validate, Mode

class TestCNN:

    @staticmethod
    def test_epoch_resume():
        args, device, checkpoint = init_pipeline(['--epoch=1'])
        train_loader, val_loader, class_labels, init_params = load_train_data(args)
        model, criterion, optimizer = load_model(args, device, checkpoint, init_params, train_loader)
        run_name, metrics = init_metrics(args, checkpoint)

        metrics.next_epoch()
        train_loss = train_and_validate(model, train_loader, optimizer, criterion,
                                        device, class_labels, metrics, Mode.TRAIN)

        assert True
