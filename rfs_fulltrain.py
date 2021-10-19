
import contextlib
from datetime import datetime
import json
import os
from skimage.color import gray2rgb
# import torch.optim as optim
# from torch.utils.data import DataLoader

from rfs_preprocessing import *
from rfs_utils import *
from rfs_models import *
import config as args


def train_ct_and_gene():
    ## PRELIMINARY SETUP ##
    # global args, device

    # Utilize GPUs for Tensor computations if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get input arguments (either from defaults or input when this function is called from terminal)
    # args = parser.parse_args()

    ## OUTPUT SETUP ##
    # Check if testing mode to see if output should be saved or not
    if not args.testing:
        # Setup output directory to save parameters/results/etc.
        out_dir = 'Output/' + str(args.modelname) + '-' + datetime.now().strftime("%Y-%m-%d-%H%M")
        out_path = os.path.join(args.datadir, out_dir)

        # Make output folder for the training run
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        # Save out parameters used for the run
        save_param_fname = os.path.join(out_path, 'parameters.txt')
        with open(save_param_fname, 'w') as f:
            with contextlib.redirect_stdout(f):
                help(args)
            # json.dump(dir(args), f, indent=2)

        # Setting up file to save out evaluation values to/load them from
        save_eval_fname = os.path.join(out_path, 'convergence.csv')

        # Setting up file to save out final values
        save_final_fname = os.path.join(out_path, 'final_results.csv')

    ## DATA LOADING ##
    if args.validation:
        # Loading the tumor images, splitting into train/valid/test based on event indicator, and setting up DataLoader
        # objects for each
        train_loader, valid_loader, test_loader = load_chol_tumor_w_gene(args.datadir, imdim=args.imdim,
                                                                         scanthresh=args.scanthresh, split=args.split,
                                                                         batch_size=args.batchsize, valid=True,
                                                                         valid_split=args.valid_split,
                                                                         seed=args.randseed)
    else:
        # Loading the tumor images, splitting into train/test based on event indicator, and setting up DataLoader
        # objects for each
        train_loader, test_loader = load_chol_tumor_w_gene(args.datadir, imdim=args.imdim, scanthresh=args.scanthresh,
                                                           split=args.split, batch_size=args.batchsize, valid=False,
                                                           seed=args.randseed)

    ## MODEL SETUP ##
    # Need number of genes for first layer of FC architecture in CholClassifier models
    num_genes = train_loader.dataset.num_genes
    # Select which model architecture to build
    if args.modelname == 'CholClassifier18':
        model = CholClassifier('18', num_genes, l2=256, l3=64, l4=16, l5=32, d1=0.19, d2=0.12).to(device)
    elif args.modelname == 'CholClassifier34':
        model = CholClassifier('34', num_genes).to(device)
    else:
        raise Exception("This function can only use models that take image and genetic data as inputs "
                        "(e.g. CholClassifier18). Please update config.py.")

    # model = select_model(args.modelname, device, num_genes=num_genes)
    #
    # if type(model) != CholClassifier:
    #     raise Exception("This function can only use models that take image and genetic data as inputs "
    #                     "(e.g. CholClassifier18). Please update config.py.")

    # Define loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learnrate)
    criterion = NegativeLogLikelihood(device)

    ## MODEL TRAINING ##
    for epoch in range(0, args.epochs):
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

        # Validation phase
        if args.validation:
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

            # Printing average loss and c-index values for the epoch
            print('Epoch: {} \t Train Loss: {:.4f} \t Val Loss: {:.4f} \t Train CI: {:.3f} \t Val CI: {:.3f}'.format(epoch,
                   coxLossMeter.avg, valLossMeter.avg, ciMeter.avg, ciValMeter.avg))

            if not args.testing:
                # Saving evaluation metrics, using average across all batches for this epoch
                save_error(train_ci=ciMeter.avg, val_ci=ciValMeter.avg,
                           coxLoss=coxLossMeter.avg, valCoxLoss=valLossMeter.avg,
                           variance=varMeter.avg, epoch=epoch, slname=save_eval_fname)

        else:
            # Validation not used, display/save only training results
            print('Epoch: {} \t Train Loss: {:.4f} \t Train CI: {:.4f}'.format(epoch, coxLossMeter.avg, ciMeter.avg))

            if not args.testing:
                # Saving evaluation metrics, using average across all batches for this epoch
                save_error(train_ci=ciMeter.avg, coxLoss=coxLossMeter.avg,
                           variance=varMeter.avg, epoch=epoch, slname=save_eval_fname)

    if not args.testing:
        plot_coxloss(save_eval_fname, model._get_name(), valid=args.validation, save=args.saveplots)
        plot_concordance(save_eval_fname, model._get_name(), valid=args.validation, save=args.saveplots)

    ## MODEL TESTING ##
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
        test_X, test_g, test_y, test_e = test_X.float().to(device), test_g.float().to(device), test_y.float().to(device), test_e.float().to(device)

        # Forward pass through the model
        test_risk_pred = model(test_X, test_g)

        # Calculate loss and evaluation metrics
        test_cox_loss = criterion(-test_risk_pred, test_y, test_e, model)
        testLossMeter.update(test_cox_loss.item(), test_y.size(0))

        test_ci = c_index(test_risk_pred, test_y, test_e)
        ciTestMeter.update(test_ci.item(), test_y.size(0))

    print('Test Loss: {:.4f} \t Test CI: {:.4f}'.format(testLossMeter.avg, ciTestMeter.avg))

    if not args.testing:
        # Save out Pytorch model as .pth file so it can be reloaded if successful
        savemodel(out_path, model)

        if args.validation:
            save_final_result(train_ci=ciMeter.avg, val_ci=ciValMeter.avg, test_ci=ciTestMeter.avg,
                              coxLoss=coxLossMeter.avg, valCoxLoss=valLossMeter.avg, testCoxLoss=testLossMeter.avg,
                              slname=save_final_fname)
        else:
            save_final_result(train_ci=ciMeter.avg, test_ci=ciTestMeter.avg,
                              coxLoss=coxLossMeter.avg, testCoxLoss=testLossMeter.avg,
                              slname=save_final_fname)


if __name__ == '__main__':
    train_ct_and_gene()
