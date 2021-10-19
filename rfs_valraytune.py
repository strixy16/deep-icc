from functools import partial
from skimage.color import gray2rgb
import torch.optim as optim
from ray import tune
from ray.tune import CLIReporter

from rfs_preprocessing import *
from rfs_utils import *
from rfs_models import *
import config as args


def train_val_ct_and_gene(config, checkpoint_dir=None, data_dir=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader, valid_loader, _ = load_chol_tumor_w_gene(data_dir,
                                                           imdim=args.imdim,
                                                           scanthresh=args.scanthresh,
                                                           split=args.split,
                                                           batch_size=args.batchsize,
                                                           valid=True,
                                                           seed=args.randseed)

    num_genes = train_loader.dataset.num_genes

    # Not using select_model here because need to pass tuning parameters
    if args.modelname == 'CholClassifier18':
        model = CholClassifier('18', num_genes, l2=config['l2'], l3=config['l3'], l4=config['l4'], l5=config['l5'],
                               d1=config['d1'], d2=config['d2'])
    elif args.modelname == 'CholClassifier34':
        model = CholClassifier('34', num_genes, l2=config['l2'], l3=config['l3'], l4=config['l4'], l5=config['l5'],
                               d1=config['d1'], d2=config['d2'])
    else:
        raise Exception('Invalid model type name. Must be CholClassifier.')

    model.to(device)

    criterion = NegativeLogLikelihood(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load.state(model_state)
        optimizer.load_state_dict(optimizer_state)

    for epoch in range(args.epochs):
        # Initialize/reset value holders for training loss, c-index, and var values
        coxLossMeter = AverageMeter()
        ciMeter = AverageMeter()
        varMeter = AverageMeter()

        # Training phase
        model.train()
        for X, g, y, e, _ in train_loader:
            # X = CT image
            # g = genetic markers
            # y = time to event
            # e = event indicator

            # Resnet models expect an RGB image, so a 3 channel version of the CT image is generated here
            # (CholClassifier contains a ResNet component, so included here as well)
            if type(model) == ResNet or type(model) == CholClassifier:
                # Convert grayscale image to rgb to generate 3 channels
                rgb_X = gray2rgb(X)
                # Reshape so channels is second value
                rgb_X = torch.from_numpy(rgb_X)
                X = torch.reshape(rgb_X, (rgb_X.shape[0], rgb_X.shape[-1], rgb_X.shape[2], rgb_X.shape[3]))

            # Convert all values to float for backprop and evaluation calculations
            X, g, y, e = X.float().to(device), g.float().to(device), y.float().to(device), e.float().to(device)

            # Forward pass through model, passing image and gene data
            risk_pred = model(X, g)

            # Calculate loss and evaluation metric
            cox_loss = criterion(-risk_pred, y, e, model)
            coxLossMeter.update(cox_loss.item(), y.size(0))

            train_ci = c_index(risk_pred, y, e)
            ciMeter.update(train_ci.item(), y.size(0))

            varMeter.update(risk_pred.var(), y.size(0))

            # Updating parameters based on forward pass
            optimizer.zero_grad()
            cox_loss.backward()
            optimizer.step()
        # End training for loop

        # Validation phase
        model.eval()
        # Initialize/reset value holders for validation loss, c-index
        valLossMeter = AverageMeter()
        ciValMeter = AverageMeter()
        for val_X, val_g, val_y, val_e, _ in valid_loader:
            # val_X = CT image
            # val_g = genetic markers
            # val_y = time to event
            # val_e = event indicator

            # ResNet models expect an RGB image, so a 3 channel version of the CT image is generated here
            # (CholClassifier contains a ResNet component, so included here as well)
            if type(model) == ResNet or type(model) == CholClassifier:
                # Convert grayscale image to rgb to generate 3 channels
                rgb_valX = gray2rgb(val_X)
                # Reshape so channels is second value
                rgb_valX = torch.from_numpy(rgb_valX)
                val_X = torch.reshape(rgb_valX,
                                      (rgb_valX.shape[0], rgb_valX.shape[-1], rgb_valX.shape[2], rgb_valX.shape[3]))

            # Convert all values to float for backprop and evaluation calculations
            val_X, val_g, val_y, val_e = val_X.float().to(device), val_g.float().to(device), val_y.to(device), val_e.to(device)

            # Forward pass through the model
            val_risk_pred = model(val_X, val_g)

            # Calculate loss and evaluation metrics
            val_cox_loss = criterion(-val_risk_pred, val_y, val_e, model)
            valLossMeter.update(val_cox_loss.item(), y.size(0))

            val_ci = c_index(val_risk_pred, val_y, val_e)
            ciValMeter.update(val_ci.item(), val_y.size(0))
        # End validation for loop

        print(tune.checkpoint_dir(epoch))
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(train_loss=coxLossMeter.avg, train_cind=ciMeter.avg,
                    val_loss=valLossMeter.avg, val_cind=ciValMeter.avg)

    print("Finished Training")
    torch.cuda.empty_cache()


def test_ct_and_gene(model, device, data_dir=None):
    _, _, test_loader = load_chol_tumor_w_gene(data_dir,
                                               imdim=args.imdim,
                                               scanthresh=args.scanthresh,
                                               split=args.split,
                                               batch_size=args.batchsize,
                                               valid=True,
                                               seed=args.randseed)

    criterion = NegativeLogLikelihood(device)
    model.eval()
    # Initialize value holders for testing loss, c-index
    testLossMeter = AverageMeter()
    ciTestMeter = AverageMeter()
    for test_X, test_g, test_y, test_e, _ in test_loader:
        # test_X = CT image
        # test_g = genetic markers
        # test_y = time to event
        # test_e = event indicator

        # ResNet models expect an RGB image, so a 3 channel version of the CT image is generated here
        # (CholClassifier contains a ResNet component, so included here as well)
        if type(model) == ResNet or type(model) == CholClassifier:
            # Convert grayscale image to rgb to generate 3 channels
            rgb_testX = gray2rgb(test_X)
            # Reshape so channels is second value
            rgb_testX = torch.from_numpy(rgb_testX)
            test_X = torch.reshape(rgb_testX,
                                   (rgb_testX.shape[0], rgb_testX.shape[-1], rgb_testX.shape[2], rgb_testX.shape[3]))

        # Convert all values to float for backprop and evaluation calculations
        test_X, test_g, test_y, test_e = test_X.float().to(device), test_g.float().to(device), test_y.float().to(
            device), test_e.float().to(device)

        # Forward pass through the model
        test_risk_pred = model(test_X, test_g)

        # Calculate loss and evaluation metrics
        test_cox_loss = criterion(-test_risk_pred, test_y, test_e, model)
        testLossMeter.update(test_cox_loss.item(), test_y.size(0))

        test_ci = c_index(test_risk_pred, test_y, test_e)
        ciTestMeter.update(test_ci.item(), test_y.size(0))

    # print('Test Loss: {:.4f} \t Test CI: {:.4f}'.format(testLossMeter.avg, ciTestMeter.avg))
    return testLossMeter.avg, ciTestMeter.avg


def main(num_samples=10, gpus_per_trial=1):
    data_dir = args.datadir
    ray_dir = os.path.join(data_dir, 'ray_results')

    config = {
        "l2": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
        "l3": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
        "l4": tune.choice([4, 8, 16, 32]),
        "l5": tune.choice([2, 4, 8, 16, 32]),
        "d1": tune.loguniform(0.1, 0.9),
        "d2": tune.loguniform(0.1, 0.9),
        "lr": tune.loguniform(1e-5, 1e-3)
    }

    reporter = CLIReporter(
        # parameter_columns=["l2", "l3", "l4", "l5", "d1", "d2", "lr"]
        metric_columns=["train_loss", "train_cind", "val_loss", "val_cind", "training_iteration"])

    result = tune.run(
        partial(train_val_ct_and_gene, data_dir=data_dir),
        resources_per_trial={"cpu": 4, "gpu": gpus_per_trial},
        config=config,  # Dictionary of hyperparameters to tune
        local_dir=ray_dir, # Where to save checkpoints
        num_samples=num_samples, # Number of trials to run
        progress_reporter=reporter # Reporter to display updates during training
    )

    torch.cuda.empty_cache()

    best_trial = result.get_best_trial("val_cind", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["val_loss"]))
    print("Best trial final validation c-index: {}".format(best_trial.last_result["val_cind"]))

    if args.modelname == 'CholClassifier18':
        best_trained_model = CholClassifier('18', 18, l2=best_trial.config['l2'], l3=best_trial.config['l3'],
                                            l4=best_trial.config['l4'], l5=best_trial.config['l5'],
                                            d1=best_trial.config['d1'], d2=best_trial.config['d2'])
    elif args.modelname == 'CholClassifier34':
        best_trained_model = CholClassifier('34', 18, l2=best_trial.config['l2'], l3=best_trial.config['l3'],
                                            l4=best_trial.config['l4'], l5=best_trial.config['l5'],
                                            d1=best_trial.config['d1'], d2=best_trial.config['d2'])
    else:
        raise Exception('Invalid model type name. Must be CholClassifier.')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_loss, test_c_index = test_ct_and_gene(best_trained_model, device, data_dir)
    print("Best trial test loss: {}".format(test_loss))
    print("Best trial test set c-index: {}".format(test_c_index))


if __name__ == "__main__":
    main(num_samples=10, gpus_per_trial=1)
