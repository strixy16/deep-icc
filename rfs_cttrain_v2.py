import contextlib
from datetime import datetime
import json
import os
import matplotlib as plt
import torch.cuda
from skimage.color import gray2rgb
# import torch.optim as optim
# from torch.utils.data import DataLoader

from rfs_preprocessing import *
from rfs_utils import *
from rfs_models import *
import config as args


def train_ct():
    ## PRELIMINARY SETUP ##
    # global args

    # Utilize GPUs for Tensor computations if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
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

        # Setting up file to save out evaluation values to/load them from
        save_eval_fname = os.path.join(out_path, 'convergence.csv')

        # Setting up file to save out final values
        save_final_fname = os.path.join(out_path, 'final_results.csv')

        # Setting up file to save out actual/predicted test label
        save_label_fname = os.path.join(out_path, 'label_comparison.csv')


    ## DATA LOADING ##
    if args.validation:
        # Loading the tumor images, splitting into train/valid/test based on event indicator, and setting up DataLoader
        # objects for each
        train_loader, valid_loader, test_loader = load_chol_tumor(args.datadir, imdim=args.imdim,
                                                                  scanthresh=args.scanthresh, split=args.split,
                                                                  batch_size=args.batchsize, valid=True,
                                                                  valid_split=args.valid_split, seed=args.randseed)
    else:
        # Loading the tumor images, splitting into train/test based on event indicator, and setting up DataLoader
        # objects for each
        train_loader, test_loader = load_chol_tumor(args.datadir, imdim=args.imdim, scanthresh=args.scanthresh,
                                                    split=args.split, batch_size=args.batchsize, valid=False,
                                                    seed=args.randseed)

    ## MODEL SETUP ##
    # Select which model architecture to build
    model = select_model(args.modelname, device)
    if type(model) != KT6Model or type(model) != DeepConvSurv or type(model) != ResNet:
        raise Exception("This function can only use models that take image data as inputs. "
                        "(e.g. KT6, DeepConvSurv, ResNet) Please update config.py.")
    # Setting optimization method and loss function
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
        for X, y, e, _ in train_loader:
            # X = CT image
            # y = time to event
            # e = event indicator
            # _ is ignoring the slice file name, not needed here

            # To view images to ensure proper loading
            # for imnum in range(0, args.batchsize):
            #     plt.imshow(X[imnum][0])
            #     plt.show()
            #     print(imnum)
            #     # Place breakpoint here to loop through images
            #     print()

            # ResNet models expect an RGB image, so a 3 channel version of the CT image is generated here
            # (CholClassifier contains a ResNet component, so included here as well)
            if type(model) == ResNet or type(model) == CholClassifier:
                # Convert grayscale image to rgb to generate 3 channels
                rgb_X = gray2rgb(X)
                # Reshape so channels is second value
                rgb_X = torch.from_numpy(rgb_X)
                X = torch.reshape(rgb_X, (rgb_X.shape[0], rgb_X.shape[-1], rgb_X.shape[2], rgb_X.shape[3]))

            # Convert all values to float for backprop and evaluation calculations
            X, y, e = X.float().to(device), y.float().to(device), e.float().to(device)

            # Forward pass through model
            risk_pred = model(X)

            # Calculate loss and evaluation metrics
            cox_loss = criterion(-risk_pred, y, e, model)
            coxLossMeter.update(cox_loss.item(), y.size(0))

            train_ci = c_index(risk_pred, y, e)
            ciMeter.update(train_ci.item(), y.size(0))

            varMeter.update(risk_pred.var(), y.size(0))

            # Updating parameters based on forward pass
            optimizer.zero_grad()
            cox_loss.backward()
            optimizer.step()
        # end train for loop

        # Validation phase
        if args.validation:
            model.eval()
            # Initialize/reset value holders for validation loss, c-index
            valLossMeter = AverageMeter()
            ciValMeter = AverageMeter()
            for val_X, val_y, val_e, _ in valid_loader:
                # val_X = CT image
                # val_y = time to event
                # val_e = event indicator
                # _ is ignoring the slice file name, not needed here

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
                val_X, val_y, val_e = val_X.float().to(device), val_y.float().to(device), val_e.float().to(device)

                # Forward pass through the model
                val_risk_pred = model(val_X)

                # Calculate loss and evaluation metrics
                val_cox_loss = criterion(-val_risk_pred, val_y, val_e, model)
                valLossMeter.update(val_cox_loss.item(), val_y.size(0))

                val_ci = c_index(val_risk_pred, val_y, val_e)
                ciValMeter.update(val_ci.item(), val_y.size(0))

            # Printing the average so you get average across all the batches for this epoch.
            print('Epoch: {} \t Train Loss: {:.4f} \t Val Loss: {:.4f} \t Train CI: {:.4f} \t Val CI: {:.4f}'.format(
                  epoch, coxLossMeter.avg, valLossMeter.avg, ciMeter.avg, ciValMeter.avg))
            # end validation for loop

            if not args.testing:
                # Saving evaluation metrics, using average across all batches for this epoch
                save_error(train_ci=ciMeter.avg, val_ci=ciValMeter.avg,
                           coxLoss=coxLossMeter.avg, valCoxLoss=valLossMeter.avg,
                           variance=varMeter.avg, epoch=epoch, slname=save_eval_fname)
            # end if testing check
        # end if validation check

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

    # TODO: add final row of average values from the AverageMeters

    ## MODEL TESTING ##
    model.eval()
    testLossMeter = AverageMeter()
    ciTestMeter = AverageMeter()
    labelComp = pd.DataFrame(columns=['Slice_File_Name', 'Actual', 'Predicted'])

    # Only going over each batch once
    for test_X, test_y, test_e, test_slice_fname in test_loader:
        # test_X = CT image
        # test_y = time to event
        # test_e = event indicator
        # test_slice_fname = name of slice file

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
        test_X, test_y, test_e = test_X.float().to(device), test_y.float().to(device), test_e.float().to(device)

        # Forward pass through the model
        test_risk_pred = model(test_X)

        # Calculate loss and evaluation metrics
        test_cox_loss = criterion(-test_risk_pred, test_y, test_e, model)
        testLossMeter.update(test_cox_loss.item(), test_y.size(0))

        test_ci = c_index(test_risk_pred, test_y, test_e)
        ciTestMeter.update(test_ci.item(), test_y.size(0))

        df_batch = pd.DataFrame(list(test_slice_fname), columns=['Slice_File_Name'])
        df_batch['Actual'] = test_y.cpu().detach().numpy()
        df_batch['Predicted'] = test_risk_pred.cpu().detach().numpy()

        labelComp = labelComp.append(df_batch)

    labelComp.sort_values(by=['Slice_File_Name'], inplace=True)
    print('Test Loss: {:.4f} \t Test CI: {:.4f}'.format(testLossMeter.avg, ciTestMeter.avg))

    if not args.testing:
        # Saving out Pytorch model as .pth file so it can be reloaded if successful
        savemodel(out_path, model)
        labelComp.to_csv(save_label_fname, index=False)

        # Saving out final train/valid/test statistics at end of training
        if args.validation:
            save_final_result(train_ci=ciMeter.avg, val_ci=ciValMeter.avg, test_ci=ciTestMeter.avg,
                              coxLoss=coxLossMeter.avg, valCoxLoss=valLossMeter.avg, testCoxLoss=testLossMeter.avg,
                              slname=save_final_fname)

        else:
            save_final_result(train_ci=ciMeter.avg, test_ci=ciTestMeter.avg,
                              coxLoss=coxLossMeter.avg, testCoxLoss=testLossMeter.avg,
                              slname=save_final_fname)


if __name__ == '__main__':
    train_ct()
