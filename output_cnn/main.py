from src import init_pipeline, load_train_data, load_model, init_metrics, train_and_validate, Mode


def main():
    args, device, checkpoint = init_pipeline()
    train_loader, val_loader, class_labels, init_params = load_train_data(args)
    model, criterion, optimizer = load_model(args, device, checkpoint, init_params, train_loader)
    run_name, metrics = init_metrics(args, checkpoint)
    metrics.add_network(model, train_loader, device)
    # visualize(model, train_loader, class_labels, device, run_name)

    # util.set_rng_state(checkpoint)
    start_epoch = metrics.epoch + 1
    for epoch in range(start_epoch, start_epoch + args.epochs):
        print(f'Epoch [{epoch}/{start_epoch + args.epochs - 1}]')
        metrics.next_epoch()
        train_loss = train_and_validate(model, train_loader, optimizer, criterion,
                                        device, class_labels, metrics, Mode.TRAIN)
        val_loss = train_and_validate(model, val_loader, optimizer, criterion,
                                      device, class_labels, metrics, Mode.VAL)

        is_best = metrics.update_best_metric(val_loss)
        # util.save_checkpoint({
        #     'model_init': init_params,
        #     'state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'rng_state': random.getstate(),
        #     'np_rng_state': np.random.get_state(),
        #     'torch_rng_state': torch.get_rng_state(),
        #     'run_name': run_name,
        #     'metric_obj': metrics.json_repr()
        # }, run_name, is_best)

    # visualize_trained(model, train_loader, class_labels, device, run_name)


if __name__ == '__main__':
    main()