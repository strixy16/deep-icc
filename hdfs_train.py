import contextlib
from datetime import datetime
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
# from pycox.evaluation import EvalSurv
from sklearn.model_selection import GroupKFold, KFold
import torch.optim as optim
from torchinfo import summary
from torch.utils.data import DataLoader, SubsetRandomSampler

from hdfs_data_loading import *
import hdfs_config as args
from hdfs_models import *


def view_images(data_loader):
    # Function to view images that are loaded in in the correct way
    for X, _, _, _ in data_loader:

        # im = plt.imshow(X[0][0], cmap='gray', vmin=-100, vmax=300)
        # plt.savefig("../Data/Output/adjusted_img.png")

        # Function assumes batch size of 32
        # Number of columns to make in subplot
        N_COL = 8
        # Number of rows to make in subplot
        N_ROWS = 4

        # fig = plt.figure() # Uncomment if you want each batch to have their own figure
        for index in range(1, N_COL*N_ROWS+1):
            # Select subplot to put current image in
            plt.subplot(N_ROWS, N_COL, index)
            # Turn off cartesian plane
            plt.axis('off')
            # Display figure with correct grayscale range
            plt.imshow(X[index-1][0], cmap='gray', vmin=-100, vmax=300)

        # fig.suptitle('HDFS Dataset - preview')
        # Display the complete figure
        plt.show()


def train_epoch(model, device, dataloader, criterion, optimizer):
    # Function to train a deep learning model for some number of epochs
    # Set model to training mode so weights are updated
    model.train()
    # Initialize loss counter
    coxLoss = 0.0
    # Initialize c-index counter
    conInd = 0.0
    # Initialize GHCI counter
    ghcInd = 0.0

    # use to confirm dataloader is passed correctly
    # view_images(dataloader)

    # Iterate over each batch in the data
    for X, t, e, _ in dataloader:
        # X = CT image
        # t = time to event
        # e = event indicator

        # Convert all data to floats and store on CPU or GPU (if available)
        X, t, e = X.float().to(device), t.float().to(device), e.float().to(device)

        # Forward pass through model
        risk_pred = model(X)

        # Calculate loss and evaluation metrics
        train_loss = criterion(-risk_pred, t, e, model)
        coxLoss += train_loss.item() * t.size(0)

        if args.USE_GH:
            train_ci = gh_c_index(risk_pred)
            conInd += train_ci * t.size(0)

        else:
            train_ci = c_index(risk_pred, t, e)
            conInd += train_ci.item() * t.size(0)

        # Updating parameters based on forward pass
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

    return coxLoss, conInd


def valid_epoch(model, device, dataloader, criterion):
    # Function to run validation on a deep learning model
    # Set model to evaluation so weights are not updated
    model.eval()
    # Initialize loss counter
    coxLoss = 0.0
    # Initialize c-index counter
    conInd = 0.0

    # Dataframe to save predictions to calculate overall instead of average c-index
    df_all_pred = pd.DataFrame(columns=['Slice_File_Name', 'Prediction', 'Time', 'Event'])

    # Iterate over each batch in the data
    for X, t, e, fname in dataloader:
        # X = CT image
        # t = time to event
        # e = event indicator

        # Convert all data to floats and store on CPU or GPU (if available)
        X, t, e = X.float().to(device), t.float().to(device), e.float().to(device)

        # Pass data forward through trained model
        risk_pred = model(X)

        # Calculate loss and evaluation metrics
        val_loss = criterion(-risk_pred, t, e, model)
        coxLoss += val_loss.item() * t.size(0)

        if args.USE_GH:
            val_ci = gh_c_index(risk_pred)
            conInd += val_ci * t.size(0)
        else:
            val_ci = c_index(risk_pred, t, e)
            conInd += val_ci * t.size(0)

        df_batch = pd.DataFrame(list(fname), columns=['Slice_File_Name'])
        df_batch['Prediction'] = risk_pred.cpu().detach().numpy()
        df_batch['Time'] = t.cpu().detach().numpy()
        df_batch['Event'] = e.cpu().detach().numpy()

        df_all_pred = pd.concat([df_all_pred, df_batch], ignore_index=True)
    
    df_all_pred.sort_values(by=['Slice_File_Name'], inplace=True)

    # Calculating c-index for entire testing set, not just per batch 
    all_pred = np.array(df_all_pred['Prediction'])
    all_time = np.array(df_all_pred['Time'])
    all_event = np.array(df_all_pred['Event'])

    if args.USE_GH:
        val_all_cind = gh_c_index(all_pred)
    else:
        val_all_cind = c_index(all_pred, all_time, all_event)

    
    return coxLoss, conInd, val_all_cind, df_all_pred


def kfold_train(data_info_path, data_img_path, out_dir_path, k=None, seed=16):
    
    # Load info spreadsheet for use in separating data into folds
    info = pd.read_csv(data_info_path)
    # Load training hdfs_dataset
    hdfs_dataset = HDFSTumorDataset(data_info_path, data_img_path, args.ORIG_IMG_DIM)

    # K-fold cross validation setup
    # Group fold keeps all slices from same patient together
    group_kfold = GroupKFold(n_splits=k)
    groups = hdfs_dataset.info.Pat_ID.to_numpy()

    foldperf = {}

    # Split data into k-folds and train/validate a model for each fold
    # for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(u_pats)))):
    # fold = 0
    for fold, (train_idx, val_idx) in enumerate(group_kfold.split(hdfs_dataset.img_fname, hdfs_dataset.time_label, groups)):
        # Output current fold number
        print('Fold {}'.format(fold + 1))
        # Select indices to use for train part of fold
        train_sampler = SubsetRandomSampler(train_idx)
        # Select indices to use for validation part of fold
        valid_sampler = SubsetRandomSampler(val_idx)

        # Setup dataloader for training portion of data
        train_loader = DataLoader(hdfs_dataset, batch_size=args.BATCH_SIZE, sampler=train_sampler)
        # Setup dataloader for validation portion of data
        valid_loader = DataLoader(hdfs_dataset, batch_size=args.BATCH_SIZE, sampler=valid_sampler, drop_last=False)

        # Create model to train for this fold
        model = select_model(args.MODEL_NAME)
        # Save model to CPU or GPU (if available)
        model.to(device)
        
        # Set optimizer
        if args.OPTIM == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args.LR, weight_decay=args.OPTIM_WEIGHT_DECAY)
        elif args.OPTIM == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=args.LR, weight_decay=args.OPTIM_WEIGHT_DECAY)
        else:
            raise Exception('Invalid optimizer. Must be SGD or Adam.')

        # Set loss function
        criterion = NegativeLogLikelihood(device, args.LOSS_WEIGHT_DECAY)

        # Initialize dictionary to save evaluation metrics and trained model for each fold
        history = {'train_loss': [], 'valid_loss': [], 'train_cind': [], 'valid_cind': [], 'model': None}
        for epoch in range(args.EPOCHS):
            # Train the model
            train_loss, train_cind = train_epoch(model, device, train_loader, criterion, optimizer)
            # validate the model
            valid_loss, valid_cind, _, _ = valid_epoch(model, device, valid_loader, criterion)

            # Get average metrics for this epoch
            train_loss = train_loss / len(train_loader.sampler)
            train_cind = train_cind / len(train_loader.sampler)
            valid_loss = valid_loss / len(valid_loader.sampler)
            valid_cind = valid_cind / len(valid_loader.sampler)

            # Display average metrics for this epoch
            print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Validation Loss:{:.3f} "
                  "AVG Training C-index:{:.2f}  AVG Validation C-index:{:.2f} ".format(epoch + 1,
                                                                                         args.EPOCHS,
                                                                                         train_loss,
                                                                                         valid_loss,
                                                                                         train_cind,
                                                                                         valid_cind))
            # Store average metrics for this epoch
            history['train_loss'].append(train_loss)
            history['valid_loss'].append(valid_loss)
            history['train_cind'].append(train_cind)
            history['valid_cind'].append(valid_cind)
        # END epoch loop

        _, _, total_valid_cind, valid_predictions = valid_epoch(model, device, valid_loader, criterion)

        # Store trained model for this fold
        history['model'] = model
        # Store data for this fold
        foldperf['fold{}'.format(fold+1)] = history
        # fold += 1
    # END k-fold loop

    # Setting up evaluation plots showing loss and c-index for each fold across all epochs
    fig, ax = plt.subplots(2, 2, figsize=(14,10))
    ax[0, 0].set_title('Training loss')
    ax[0, 1].set_title('Validation loss')
    ax[1, 0].set_title('Training c-index')
    ax[1, 1].set_title('Validation c-index')
    tloss_plot = plt.subplot(2, 2, 1)
    teloss_plot = plt.subplot(2, 2, 2)
    tcind_plot = plt.subplot(2, 2, 3)
    tecind_plot = plt.subplot(2, 2, 4)


    # Calculate average performance of each fold (finding mean from all epochs)
    trainl_f, validl_f, trainc_f, validc_f = [], [], [], []
    for f in range(1,args.K+1):
        # Plotting metrics across epochs of fold f
        tloss_plot.plot(foldperf['fold{}'.format(f)]['train_loss'], label='Fold {}'.format(f))
        teloss_plot.plot(foldperf['fold{}'.format(f)]['valid_loss'], label='Fold {}'.format(f))
        tcind_plot.plot(foldperf['fold{}'.format(f)]['train_cind'], label='Fold {}'.format(f))
        tecind_plot.plot(foldperf['fold{}'.format(f)]['valid_cind'], label='Fold {}'.format(f))


        # Average Train loss for fold f
        trainl_f.append(np.mean(foldperf['fold{}'.format(f)]['train_loss']))
        # Average validation loss for fold f
        validl_f.append(np.mean(foldperf['fold{}'.format(f)]['valid_loss']))
        # Average training c-index for fold f
        trainc_f.append(np.mean(foldperf['fold{}'.format(f)]['train_cind']))
        # Average validation c-index for fold f
        validc_f.append(np.mean(foldperf['fold{}'.format(f)]['valid_cind']))

    # Setting up legend for evaluation plots
    labels = ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"]
    fig.legend([tloss_plot, teloss_plot, tcind_plot, tecind_plot], labels=labels, loc="upper right")
    # Setting overall title
    fig.suptitle("Evaluation metrics for k-fold cross validation")

    if not args.DEBUG:
        # Save evaluation plots figure
        plt.savefig(os.path.join(out_dir_path, 'eval_plots.png'))
    else:
        plt.show()

    # Output performance for this run
    print('Performance of {} fold cross validation'.format(args.K))
    # Print the average loss and c-index for training and validation across all folds (Model performance)
    print("Average Training Loss: {:.3f} \t Average Validation Loss: {:.3f} \t "
          "Average Training C-Index: {:.2f} \t Average Validation C-Index: {:.2f}".format(np.mean(trainl_f),
                                                                                          np.mean(validl_f),
                                                                                          np.mean(trainc_f),
                                                                                          np.mean(validc_f)))
    # Find fold with best performance
    best_avg_loss = np.amin(validl_f)
    best_avg_cind = np.amax(validc_f)
    # fold_w_best_loss = np.where(validl_f == best_loss)
    # Get index of fold with best c-index
    fold_w_best_avg_cind = np.where(validc_f == best_avg_cind)
    # Get actual fold number (folds start at 1)
    best_fold = int(fold_w_best_avg_cind[0]) + 1
    # Get model from the best fold
    best_model = foldperf['fold{}'.format(best_fold)]['model']
    tr_final_loss = foldperf['fold{}'.format(best_fold)]['train_loss'][-1]
    tr_final_cind = foldperf['fold{}'.format(best_fold)]['train_cind'][-1]
    val_final_loss = foldperf['fold{}'.format(best_fold)]['valid_loss'][-1]
    val_final_cind = foldperf['fold{}'.format(best_fold)]['valid_cind'][-1]

    # Print the best fold's final results
    print("Performance of best fold, {}:".format(best_fold))
    print("Best Training Loss: {:.3f} \t Best Validation Loss: {:.3f} \t"
          "Best Training C-index: {:.2f} \t Best Validation C-index: {:.2f}".format(tr_final_loss,
                                                                                   val_final_loss,
                                                                                   tr_final_cind,
                                                                                   val_final_cind))

    with open(os.path.join(out_dir_path, 'k_fold_results.txt'), 'w') as out_file:
            out_file.write("Performance of {} fold cross validation \n".format(args.K))
            out_file.write("Average Training Loss: {:.3f} \t Average Validation Loss: {:.3f}\nAverage Training C-Index: {:.2f} \t Average Validation C-Index: {:.2f}\n".format(np.mean(trainl_f), np.mean(validl_f), np.mean(trainc_f), np.mean(validc_f)))
            out_file.write('\n')
            out_file.write("Performance of best fold, {}: \n".format(best_fold))
            out_file.write("Best Training Loss: {:.3f} \t Best Validation Loss: {:.3f} \n"
                           "Best Training C-index: {:.2f} \t Best Validation C-index: {:.2f} \n".format(tr_final_loss,
                                                                                   val_final_loss,
                                                                                   tr_final_cind,
                                                                                   val_final_cind))
            out_file.write("\n")

    # Return model, loss, and c-ind from the best fold
    return best_model, tr_final_loss, tr_final_cind, val_final_loss, val_final_cind


def test_model(model, data_info_path, data_img_path, device):
    test_dataset = HDFSTumorDataset(data_info_path, data_img_path, args.ORIG_IMG_DIM)
    test_loader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE, shuffle=True, drop_last=False)

    model.to(device)
    model.eval()

    criterion = NegativeLogLikelihood(device)

    # test_loss, test_cind, test_predictions = valid_epoch(model, device, test_loader, criterion, test=True)

    # Function to run validation on a deep learning model
    # Set model to evaluation so weights are not updated
    model.eval()
    # Initialize loss counter
    coxLoss = 0.0
    # Initialize c-index counter
    conInd = 0.0

    # If using for testing, want to save predictions
    df_all_pred = pd.DataFrame(columns=['Slice_File_Name', 'Prediction', 'Time', 'Event'])

    # Iterate over each batch in the data
    for X, t, e, fname in test_loader:
        # X = CT image
        # t = time to event
        # e = event indicator

        # Convert all data to floats and store on CPU or GPU (if available)
        X, t, e = X.float().to(device), t.float().to(device), e.float().to(device)

        # Pass data forward through trained model
        risk_pred = model(X)

        # Calculate loss and evaluation metrics
        test_loss = criterion(-risk_pred, t, e, model)
        coxLoss += test_loss.item() * t.size(0)

        # Moved c-index calculation to end when all predictions are available
        
        df_batch = pd.DataFrame(list(fname), columns=['Slice_File_Name'])
        df_batch['Prediction'] = risk_pred.cpu().detach().numpy()
        df_batch['Time'] = t.cpu().detach().numpy()
        df_batch['Event'] = e.cpu().detach().numpy()

        df_all_pred = pd.concat([df_all_pred, df_batch], ignore_index=True)
    # end batch loop
    
    df_all_pred.sort_values(by=['Slice_File_Name'], inplace=True)

    # Calculating c-index for entire testing set, not just per batch 
    all_pred = np.array(df_all_pred['Prediction'])
    all_time = np.array(df_all_pred['Time'])
    all_event = np.array(df_all_pred['Event'])

    if args.USE_GH:
        test_cind = gh_c_index(all_pred)
    else:
        test_cind = c_index(all_pred, all_time, all_event)

    # Get average metrics for test epoch
    test_loss = coxLoss / len(test_loader.sampler)

    print("Testing loss: {:.3f} \t Testing c-index: {:.2f}".format(test_loss, test_cind))

    return test_loss, test_cind, df_all_pred


def train_main():
    """
    Main function to run if training a model from scratch
    """
    # Output setup
    out_dir = 'Output/' + args.MODEL_NAME + '/' + datetime.now().strftime("%Y_%m_%d_%H%M") + "_train"
    out_path = os.path.join(args.DATA_DIR, out_dir)

    if not args.DEBUG:
        os.makedirs(out_path)

        # Save out parameters used for the run
        save_param_fname = os.path.join(out_path, 'parameters.txt')
        with open(save_param_fname, 'w') as f:
            with contextlib.redirect_stdout(f):
                help(args)

    train_info_path = os.path.join(args.DATA_DIR, args.TRAIN_LABEL_FILE)
    test_info_path = os.path.join(args.DATA_DIR, args.TEST_LABEL_FILE)

    train_img_path = os.path.join(args.DATA_DIR, args.IMG_LOC_PATH, 'train/')
    test_img_path = os.path.join(args.DATA_DIR, args.IMG_LOC_PATH, 'test/')

    best_model, train_loss, train_cind, valid_loss, valid_cind = kfold_train(train_info_path, train_img_path, out_path,
                                                                             k=args.K, seed=args.SEED)
    torch.cuda.empty_cache()

    test_loss, test_cind, test_predictions = test_model(best_model, test_info_path, test_img_path, device)

    # Save model results
    model_stats = summary(best_model, input_size=(args.BATCH_SIZE, 1, args.ORIG_IMG_DIM, args.ORIG_IMG_DIM))
    summary_str = str(model_stats)

    if not args.DEBUG:
        # Add testing results to results.txt file made in kfold_train
        with open(os.path.join(out_path, 'k_fold_results.txt'), 'a') as kfold_file:
            kfold_file.write("Testing Results of Best Fold: \n")
            kfold_file.write("Testing Loss: {:.3f} \t Testing C-Index: {:.2f}\n\n".format(test_loss, test_cind))

        # Save summary of model
        with open(os.path.join(out_path, 'model_summary.txt'), 'w') as out_file:
            out_file.write(summary_str)

        # Save best model
        model_file_name = 'k_cross_' + args.MODEL_NAME + '.pt'
        torch.save(best_model, os.path.join(out_path, model_file_name))

        # Saving predictions made for test data
        test_predictions.to_csv(os.path.join(out_path, 'test_predictions.csv'), index=False)

    print("Completed model training/testing")


def load_main():
    """
    Main function if loading a pre-trained model to just run testing on
    """
    # Output setup
    out_dir = 'Output/' + args.MODEL_NAME + '/' + datetime.now().strftime("%Y_%m_%d_%H%M") + "_load"
    out_path = os.path.join(args.DATA_DIR, out_dir)

    if not args.DEBUG:
        os.makedirs(out_path)

        # Save out parameters used for the run
        save_param_fname = os.path.join(out_path, 'parameters.txt')
        with open(save_param_fname, 'w') as f:
            with contextlib.redirect_stdout(f):
                help(args)

    test_info_path = os.path.join(args.DATA_DIR, args.TEST_LABEL_FILE)
    test_img_path = os.path.join(args.DATA_DIR, args.IMG_LOC_PATH, str(args.ORIG_IMG_DIM), 'test/')

    load_model = torch.load(args.LOAD_MODEL_PATH)

    test_loss, test_cind, test_predictions = test_model(load_model, test_info_path, test_img_path, device)

    if not args.DEBUG:
        # Add testing results to results.txt file made in kfold_train
        with open(os.path.join(out_path, 'k_fold_results.txt'), 'a') as kfold_file:
            kfold_file.write("Testing Results of Loaded Model: \n")
            kfold_file.write("Testing Loss: {:.3f} \t Testing C-Index: {:.2f}\n\n".format(test_loss, test_cind))

        # Saving predictions made for test data
        test_predictions.to_csv(os.path.join(out_path, 'test_predictions.csv'), index=False)

    print("Completed loaded model testing")


if __name__ == '__main__':
    # Preliminaries
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Random setup
    random.seed(args.SEED)
    torch.manual_seed(args.SEED)
    torch.cuda.manual_seed(args.SEED)
    np.random.seed(args.SEED)

    if args.TRAIN_MODE:
        train_main()
    elif args.LOAD_MODE:
        load_main()
    else:
        raise Exception("No mode selected. Choose TRAIN or LOAD MODE in hdfs_config")

