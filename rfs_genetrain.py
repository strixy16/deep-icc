import argparse
import contextlib
from datetime import datetime
import json
from sklearn.model_selection import train_test_split
import os
# import optuna
# from optuna.trial import TrialState
# import pickle
# from sklearn.model_selection import KFold
from skimage.color import gray2rgb
# import torch.optim as optim
# from torch.utils.data import DataLoader

from rfs_preprocessing import *
from rfs_utils import *
from rfs_models import *
import config as args


def train_gene():
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

        # Setting up file to save out evaluation values to/load them from
        save_eval_fname = os.path.join(out_path, 'convergence.csv')

        # Setting up file to save out final values
        save_final_fname = os.path.join(out_path, 'final_results.csv')

    ## DATA LOADING ##
    genomic_file = os.path.join(args.datadir, 'labelled_Genomic_Data.xlsx')

    genes_and_labels = pd.read_excel(genomic_file)
    # Removing ScoutID so setup is proper for Survival Dataset generation
    # genes_and_labels.drop(columns=['ScoutID'], inplace=True)

    test_split = 1 - args.split
    train_genes, test_genes = train_test_split(genes_and_labels, test_size=test_split, random_state=args.randseed,
                                              shuffle=True)

    train_dataset = GeneSurvDataset(train_genes)
    test_dataset = GeneSurvDataset(test_genes)

    train_loader = DataLoader(train_dataset, batch_size=args.batchsize)
    test_loader = DataLoader(test_dataset, batch_size=args.batchsize)

    ## MODEL SETUP ##
    num_genes = train_loader.dataset.num_genes
    model = select_model(args.modelname, device, num_genes=num_genes)
    if type(model) != DeepSurvGene:
        raise Exception("This function can only use models that take gene data as inputs."
                        "(e.g. DeepSurvGene) Please update config.py.")

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
        for g, y, e in train_loader:
            # g = genetic markers
            # y = time to event
            # e = event indicator

            # Convert all values to float for backprop and evaluation calculations
            g, y, e = g.float().to(device), y.float().to(device), e.float().to(device)

            # Forward pass thruogh model
            risk_pred = model(g)

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

        print('Epoch: {} \t Train Loss: {:.4f} \t Train CI: {:.4f}'.format(epoch, coxLossMeter.avg, ciMeter.avg))

        if not args.testing:
            # Saving evaluation metrics, using average across all batches for this epoch
            save_error(train_ci=ciMeter.avg, coxLoss=coxLossMeter.avg,
                       variance=varMeter.avg, epoch=epoch, slname=save_eval_fname)

    if not args.testing:
        plot_coxloss(save_eval_fname, model._get_name(), save=args.saveplots)
        plot_concordance(save_eval_fname, model._get_name(), save=args.saveplots)


    ## MODEL TESTING ##
    model.eval()
    # Initialize value holders for testing loss, c-index
    testLossMeter = AverageMeter()
    ciTestMeter = AverageMeter()

    for test_g, test_y, test_e in test_loader:
        # test_g = genetic markers
        # test_y = time to event
        # test_e = event indicator

        # Convert all values to float for backprop and evaluation calculations
        test_g, test_y, test_e = test_g.float().to(device), test_y.float().to(device), test_e.float().to(device)

        # Forward pass through the model
        test_risk_pred = model(test_g)

        # Calculate loss and evaluation metrics
        test_cox_loss = criterion(-test_risk_pred, test_y, test_e, model)
        testLossMeter.update(test_cox_loss.item(), test_y.size(0))

        test_ci = c_index(test_risk_pred, test_y, test_e)
        ciTestMeter.update(test_ci.item(), test_y.size(0))

    print('Test Loss: {:.4f} \t Test CI: {:.4f}'.format(testLossMeter.avg, ciTestMeter.avg))

    if not args.testing:
        # Save out Pytorch model as .pth file so it can be reloaded if successful
        savemodel(out_path, model)

        save_final_result(train_ci=ciMeter.avg, test_ci=ciTestMeter.avg,
                          coxLoss=coxLossMeter.avg, testCoxLoss=testLossMeter.avg,
                          slname=save_final_fname)


if __name__ == '__main__':
    train_gene()
