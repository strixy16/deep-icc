import contextlib
from datetime import datetime
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
# from pycox.evaluation import EvalSurv
from sklearn.model_selection import GroupKFold, KFold
import torch.optim as optim
from torchinfo import summary
from torch.utils.data import DataLoader, SubsetRandomSampler, RandomSampler

from hdfs_data_loading import *
from hdfs_models import *

"""SET THIS AS HDFS_CONFIG FOR TUMORS AND HDFS_LIVER_CONFIG FOR LIVER IMAGES"""
# import hdfs_config as args
import hdfs_liver_config as args


def view_images(data_loader):
    # Function to view images that are loaded in in the correct way
    for X, _, _, _ in data_loader:

        # im = plt.imshow(X[0][0], cmap='gray', vmin=-100, vmax=300)
        # plt.savefig("../Data/Output/adjusted_img.png")

        # Function assumes batch size of 32
        # Number of columns to make in subplot
        N_COL = 4
        # Number of rows to make in subplot
        N_ROWS = 4

        fig = plt.figure(figsize=(50,50)) # Uncomment if you want each batch to have their own figure
        # This doesn't handle batches that aren't full size (last batch)
        for index in range(1, N_COL*N_ROWS+1):
            # Select subplot to put current image in
            plt.subplot(N_ROWS, N_COL, index)
            # Turn off cartesian plane
            plt.axis('off')
            # Display figure with correct grayscale range
            plt.imshow(X[index-1][0], cmap='gray', vmin=-100, vmax=300)

        # fig.suptitle('HDFS Dataset - preview')
        # Display the complete figure
        plt.savefig("/Data/liver_batch1_slices")


def train_epoch(model, device, dataloader, criterion, optimizer):
    # Function to train a deep learning model for some number of epochs
    # Set model to training mode so weights are updated
    model.train()
    # Initialize loss counter
    coxLoss = 0.0
    # Initialize c-index counter
    conInd = 0.0
    # Initialize C-statistic counter
    cstat = 0.0

    # use to confirm dataloader is passed correctly
    # view_images(dataloader)

    # Dataframe to save predictions to export at the end from best fold
    df_all_pred = pd.DataFrame(columns=['Slice_File_Name', 'Prediction', 'Time', 'Event'])

    # Iterate over each batch in the data
    for X, t, e, fname in dataloader:
        # X = CT image
        # t = time to event
        # e = event indicator

        # Convert all data to floats and store on CPU or GPU (if available)
        X, t, e = X.float().to(device), t.float().to(device), e.float().to(device)

        # Forward pass through model
        risk_pred = model(X)

        if torch.any(torch.isnan(risk_pred)):
            print("NaNs predicted by model.")

        # Calculate loss and evaluation metrics
        train_loss = criterion(-risk_pred, t, e, model)
        coxLoss += train_loss.item() * t.size(0)

        train_ci = c_index(risk_pred, t, e)
        conInd += train_ci.item() * t.size(0)
        train_cstat = uno_c_statistic(t, e, t, e, risk_pred)
        cstat += train_cstat * t.size(0)

        # Store risk predictions for criterion and c-index calculation at end of batch loop
        df_batch = pd.DataFrame(list(fname), columns=['Slice_File_Name'])
        df_batch['Prediction'] = risk_pred.cpu().detach().numpy()
        df_batch['Time'] = t.cpu().detach().numpy()
        df_batch['Event'] = e.cpu().detach().numpy()

        df_all_pred = pd.concat([df_all_pred, df_batch], ignore_index=True)

        # Updating parameters based on forward pass
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

    # Sort predictions so all slices are next to each other for easier human reading/comparison
    df_all_pred.sort_values(by=['Slice_File_Name'], inplace=True)

    return coxLoss, conInd, cstat, df_all_pred


def valid_epoch(model, device, valid_dataloader, criterion, train_dataloader=None):
    """
    Validation epoch for training a pytorch model

    Args:
        model: pytorch model (nn.Module), trained model to run testing data through
        device: torch.device, device to use for testing (GPU or CPU)
        dataloader: torch.Dataloader, loader object for the validation data
        criterion: loss function to use for evaluatoin
    """
    # Function to run validation on a deep learning model
    # Set model to evaluation so weights are not updated
    model.eval()
    # Initialize loss counter
    coxLoss = 0.0

    # Dataframe to save predictions to calculate overall instead of average c-index
    df_all_pred = pd.DataFrame(columns=['Slice_File_Name', 'Prediction', 'Time', 'Event'])

    # Dataframe to get time and event labels from training set for Uno c-statistic calculation
    df_train_labels = pd.DataFrame(columns=['Time', 'Event'])

    # Iterate over each batch in the data
    for X, t, e, fname in valid_dataloader:
        # X = CT image
        # t = time to event
        # e = event indicator

        # Convert all data to floats and store on CPU or GPU (if available)
        X, t, e = X.float().to(device), t.float().to(device), e.float().to(device)

        # Pass data forward through trained model
        risk_pred = model(X)

        if torch.any(torch.isnan(risk_pred)):
            # view_images(dataloader)
            print("NaN predicted.")

        # Store risk predictions for criterion and c-index calculation at end of batch loop
        df_batch = pd.DataFrame(list(fname), columns=['Slice_File_Name'])
        df_batch['Prediction'] = risk_pred.cpu().detach().numpy()
        df_batch['Time'] = t.cpu().detach().numpy()
        df_batch['Event'] = e.cpu().detach().numpy()

        df_all_pred = pd.concat([df_all_pred, df_batch], ignore_index=True)
    # end batch loop

    # Get training dataset labels for use in uno_c_statistic function
    for _, train_t, train_e, fname in train_dataloader:
        df_train_batch = pd.DataFrame(list(train_t.cpu().detach().numpy()), columns=['Time'])
        df_train_batch['Event'] = train_e.cpu().detach().numpy()

        df_train_labels = pd.concat([df_train_labels, df_train_batch], ignore_index=True)

    # Sort predictions so all slices are next to each other for easier human reading/comparison
    df_all_pred.sort_values(by=['Slice_File_Name'], inplace=True)

    # Calculating c-index for entire validation set, not just per batch
    # Getting prediction, time, and event separated again to pass into c_index and criterion functions
    all_pred = np.array(df_all_pred['Prediction'], dtype=float)
    all_time = np.array(df_all_pred['Time'], dtype=float)
    all_event = np.array(df_all_pred['Event'], dtype=float)

    train_time = np.array(df_train_labels['Time'], dtype=float)
    train_event = np.array(df_train_labels['Event'], dtype=float)

    # Harrell's C-index calculation
    val_all_cind = c_index(all_pred, all_time, all_event)
    # Uno's C-statistic calculation
    val_all_cstat = uno_c_statistic(train_time, train_event, all_time, all_event, all_pred)

    # Calculating criterion for entire validation set, not just per batch
    all_pred_T = torch.from_numpy(-all_pred).to(device)
    # Time gets transposed in the criterion calculation, so needs to be 2D (Second dimension is just 1)
    all_time_T = torch.reshape(torch.from_numpy(all_time), (all_time.shape[0], 1)).to(device)
    all_event_T = torch.from_numpy(all_event).to(device)

    val_loss = criterion(all_pred_T, all_time_T, all_event_T, model)

    return val_loss.item(), val_all_cind, val_all_cstat, df_all_pred


def kfold_train(data_info_path, data_img_path, out_dir_path, k=None):
    """
    Function to train a pytorch model using k-fold cross validation

    Args:
        data_info_path: str, file path to label spreadsheet for training data
        data_img_path: str, file path to directory containing training image data
        out_dir_path: str, file path to output directory for saving results
        k: int, number of folds to use

    Returns:
    best_model, tr_final_loss, tr_final_cind, val_final_loss, val_final_cind
        best_model: trained model, chosen as the fold with the highest final c-index
        tr_final_loss: float, final training loss value from best model
        tr_final_cind: float, final training c-index value from best model
        val_final_loss: float, final validation loss value from best model
        val_final_cind: float, final validation c-index value from best model
    """

    # Load training hdfs_dataset
    hdfs_dataset = HDFSTumorDataset(data_info_path, data_img_path, args.ORIG_IMG_DIM, transform_list=args.TRANSFORM_LIST)

    # K-fold cross validation setup
    # Group fold keeps all slices from same patient together
    group_kfold = GroupKFold(n_splits=k)
    # Set groups as patient ID
    groups = hdfs_dataset.info.Pat_ID.to_numpy()

    # Dictionary to store performance statistics and final model generated by each fold
    foldperf = {}
    # Dataframe to store the final performance values of each fold
    finalperf = pd.DataFrame(columns=['Fold', 'Final_Train_Loss', 'Final_Train_C-index', 'Final_Train_UnoC', 'Final_Valid_Loss', 'Final_Valid_C-index', 'Final_Valid_UnoC'])

    # Dataframe to store final validation predictions from each fold to perform final c-index calculation
    all_fold_valid_predictions = pd.DataFrame(columns=['Slice_File_Name', 'Prediction', 'Time', 'Event'])

    # Tracking current best fold to store the predictions from that fold
    best_fold_valid_loss = 10000
    best_fold_train_preds = None
    best_fold_valid_preds = None

    # Split data into k-folds and train/validate a model for each fold
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

        # view_images(train_loader)

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
        history = {'train_loss': [], 'valid_loss': [], 'train_cind': [], 'valid_cind': [], 'train_unoc': [], 'valid_unoc': [], 'model': None}
        for epoch in range(args.EPOCHS):
            # Train the model
            train_loss, train_cind, train_unoc, train_predictions = train_epoch(model, device, train_loader, criterion, optimizer)
            # validate the model
            valid_loss, valid_cind, valid_unoc, valid_predictions = valid_epoch(model, device, valid_loader, criterion, train_loader)

            # Get average training metrics for this epoch (loss and cind were calculated per batch)
            train_loss = train_loss / len(train_loader.sampler)
            train_cind = train_cind / len(train_loader.sampler)
            train_unoc = train_unoc / len(train_loader.sampler)

            # Display metrics for this epoch
            print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Validation Loss:{:.3f} "
                  "Training C-index:{:.3f}  Validation C-index:{:.3f} "
                  "Training Uno C:{:.3f}  Validation Uno C:{:.3f}".format(epoch + 1,
                                                                               args.EPOCHS,
                                                                               train_loss,
                                                                               valid_loss,
                                                                               train_cind,
                                                                               valid_cind,
                                                                               train_unoc,
                                                                               valid_unoc))
            # Store average metrics for this epoch
            history['train_loss'].append(train_loss)
            history['train_cind'].append(train_cind)
            history['train_unoc'].append(train_unoc)
            # Validation metrics not averaged, but calculated for the entire validation dataset at once
            history['valid_loss'].append(valid_loss)
            history['valid_cind'].append(valid_cind)
            history['valid_unoc'].append(valid_unoc)
        # END epoch loop

        final_valid_loss = history['valid_loss'][-1]
        if final_valid_loss < best_fold_valid_loss:
            best_fold_valid_loss = final_valid_loss
            best_fold_train_preds = train_predictions
            best_fold_valid_preds = valid_predictions

        all_fold_valid_predictions = pd.concat([all_fold_valid_predictions, valid_predictions], ignore_index=True)

        # Store trained model for this fold
        history['model'] = model
        # Store data for this fold
        foldperf['fold{}'.format(fold+1)] = history

        # Get metrics from final epoch
        finalperf.loc[len(finalperf.index)] = [fold+1, train_loss, train_cind, train_unoc, valid_loss, valid_cind, valid_unoc]

    # END k-fold loop
    finalperf = finalperf.astype({'Fold': int})

     # Calculating c-index for entire testing set, not just per batch
    all_pred = np.array(all_fold_valid_predictions['Prediction'])
    all_time = np.array(all_fold_valid_predictions['Time'])
    all_event = np.array(all_fold_valid_predictions['Event'])
    val_all_cind = c_index(all_pred, all_time, all_event)
    val_all_unoc = uno_c_statistic(all_time, all_event, all_time, all_event, all_pred)


    # Setting up evaluation plots showing loss and c-index for each fold across all epochs
    fig, ax = plt.subplots(3, 2, figsize=(14,10))
    ax[0, 0].set_title('Training loss')
    ax[0, 1].set_title('Validation loss')
    ax[1, 0].set_title('Training c-index')
    ax[1, 1].set_title('Validation c-index')
    ax[2, 0].set_title('Training Uno C-statistic')
    ax[2, 1].set_title('Validation Uno C-statistic')
    tloss_plot = plt.subplot(3, 2, 1)
    valloss_plot = plt.subplot(3, 2, 2)
    tcind_plot = plt.subplot(3, 2, 3)
    valcind_plot = plt.subplot(3, 2, 4)
    tunoc_plot = plt.subplot(3, 2, 5)
    valunoc_plot = plt.subplot(3, 2, 6)


    # Calculate average performance of each fold (finding mean from all epochs)
    trainl_f, validl_f, trainc_f, validc_f, trainunoc_f, validunoc_f = [], [], [], [], [], []
    for f in range(1,args.K+1):
        # Plotting metrics across epochs of fold f
        tloss_plot.plot(foldperf['fold{}'.format(f)]['train_loss'], label='Fold {}'.format(f))
        valloss_plot.plot(foldperf['fold{}'.format(f)]['valid_loss'], label='Fold {}'.format(f))
        tcind_plot.plot(foldperf['fold{}'.format(f)]['train_cind'], label='Fold {}'.format(f))
        valcind_plot.plot(foldperf['fold{}'.format(f)]['valid_cind'], label='Fold {}'.format(f))
        tunoc_plot.plot(foldperf['fold{}'.format(f)]['train_unoc'], label='Fold {}'.format(f))
        valunoc_plot.plot(foldperf['fold{}'.format(f)]['valid_unoc'], label='Fold {}'.format(f))

        # Average Train loss for fold f
        trainl_f.append(np.mean(foldperf['fold{}'.format(f)]['train_loss']))
        # Average validation loss for fold f
        validl_f.append(np.mean(foldperf['fold{}'.format(f)]['valid_loss']))
        # Average training c-index for fold f
        trainc_f.append(np.mean(foldperf['fold{}'.format(f)]['train_cind']))
        # Average validation c-index for fold f
        validc_f.append(np.mean(foldperf['fold{}'.format(f)]['valid_cind']))
        # Average training Uno C-statistic for fold f
        trainunoc_f.append(np.mean(foldperf['fold{}'.format(f)]['train_unoc']))
        # Average validation Uno C-statistic for fold f
        validunoc_f.append(np.mean(foldperf['fold{}'.format(f)]['valid_unoc']))

    # Setting up legend for evaluation plots
    labels = ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"]
    fig.legend([tloss_plot, valloss_plot, tcind_plot, valcind_plot, tunoc_plot, valunoc_plot], labels=labels, loc="upper right")
    # Setting overall title
    fig.suptitle("Evaluation metrics for k-fold cross validation")

    if not args.DEBUG:
        # Save evaluation plots figure
        plt.savefig(os.path.join(out_dir_path, 'eval_plots.png'))
    else:
        plt.show()


    # Find fold with best performance
    # Get index of fold with best c-index
    # best_fold = finalperf['Final_Valid_C-index'].idxmax() + 1

    # Choose fold with highest valid Uno C-statistic as best
    best_fold = finalperf['Final_Valid_Loss'].idxmin() + 1
    # Get model from the best fold
    best_model = foldperf['fold{}'.format(best_fold)]['model']
    tr_final_loss = foldperf['fold{}'.format(best_fold)]['train_loss'][-1]
    tr_final_cind = foldperf['fold{}'.format(best_fold)]['train_cind'][-1]
    tr_final_unoc = foldperf['fold{}'.format(best_fold)]['train_unoc'][-1]
    val_final_loss = foldperf['fold{}'.format(best_fold)]['valid_loss'][-1]
    val_final_cind = foldperf['fold{}'.format(best_fold)]['valid_cind'][-1]
    val_final_unoc = foldperf['fold{}'.format(best_fold)]['valid_unoc'][-1]


    if args.DEBUG:
        # Output performance for this run
        print('\nPerformance of {} fold cross validation'.format(args.K))
        print('Performance of each fold:')
        print(finalperf)
        print('\n')
        print('Averaged Final Performance of all folds:\n')
        print("Average Final Training Loss: {:.3f} \tAverage Final Validation Loss: {:.3f} \n"
            "Average Final Training C-Index: {:.3f} \tAverage Final Validation C-Index: {:.3f}\n"
            "Average Final Training Uno C-stat: {:.3f} \tAverage Final Validation Uno C-stat: {:.3f}\n".format(finalperf['Final_Train_Loss'].mean(),
                                                                                                        finalperf['Final_Valid_Loss'].mean(),
                                                                                                        finalperf['Final_Train_C-index'].mean(),
                                                                                                        finalperf['Final_Valid_C-index'].mean(),
                                                                                                        finalperf['Final_Train_UnoC'].mean(),
                                                                                                        finalperf['Final_Valid_UnoC'].mean()))
        print('Averaged Overall Performance (across epochs and folds):')
        # Print the average loss and c-index for training and validation across all folds (Model performance)
        print("Average Training Loss: {:.3f} \tAverage Validation Loss: {:.3f} \n"
            "Average Training C-Index: {:.3f} \tAverage Validation C-Index: {:.3f} \n"
            "Average Training Uno C-stat: {:.3f} \tAverage Validation Uno C-stat: {:.3f}\n".format(np.mean(trainl_f),
                                                                                    np.mean(validl_f),
                                                                                    np.mean(trainc_f),
                                                                                    np.mean(validc_f),
                                                                                    np.mean(trainunoc_f),
                                                                                    np.mean(validunoc_f)))

        print("Validation C-index for predictions from all fold models: {:.3f}\n".format(val_all_cind))
        print("Validation Uno C-stat for predictions from all fold models: {:.3f}\n".format(val_all_unoc))

        # Print the best fold's final results
        print("Performance of best fold, {}:\n".format(best_fold))
        print("Best Training Loss: {:.3f} \tBest Validation Loss: {:.3f} \n"
              "Best Training C-index: {:.3f} \tBest Validation C-index: {:.3f}\n"
              "Best Training Uno C-stat: {:.3f} \tBest Validation Uno C-stat: {:.3f}\n".format(tr_final_loss,
                                                                                val_final_loss,
                                                                                tr_final_cind,
                                                                                val_final_cind,
                                                                                tr_final_unoc,
                                                                                val_final_unoc))

    if not args.DEBUG:
        # Save performance for this run
        with open(os.path.join(out_dir_path, 'k_fold_results.txt'), 'w') as out_file:
                out_file.write("Performance of {} fold cross validation\n".format(args.K))
                out_file.write("Performance of each fold:\n")
                finalperf.to_csv(out_file, mode='a', sep="\t", index=False)

                out_file.write('\n')
                out_file.write('Averaged Final Performance of all folds:\n')
                out_file.write("Average Final Training Loss: {:.3f} \tAverage Final Validation Loss: {:.3f} \n"
                               "Average Final Training C-Index: {:.3f} \tAverage Final Validation C-Index: {:.3f}\n"
                               "Average Final Training Uno C-stat: {:.3f} \tAverage Final Validation Uno C-stat: {:.3f}\n".format(
                                   finalperf['Final_Train_Loss'].mean(), finalperf['Final_Valid_Loss'].mean(),
                                    finalperf['Final_Train_C-index'].mean(), finalperf['Final_Valid_C-index'].mean(),
                                    finalperf['Final_Train_UnoC'].mean(), finalperf['Final_Valid_UnoC'].mean()))
                out_file.write('\n')
                out_file.write('Averaged Overall Performance (across epochs and folds):\n')
                out_file.write("Average Training Loss: {:.3f} \tAverage Validation Loss: {:.3f}\n"
                               "Average Training C-Index: {:.3f} \tAverage Validation C-Index: {:.3f}\n"
                               "Average Training Uno C-stat: {:.3f} \tAverage Validation Uno C-stat: {:.3f}\n".format(
                                   np.mean(trainl_f), np.mean(validl_f), np.mean(trainc_f), np.mean(validc_f), np.mean(trainunoc_f), np.mean(validunoc_f)))
                out_file.write('\n')
                out_file.write("Validation C-index for predictions from all fold models: {:.3f}".format(val_all_cind))
                out_file.write("\nValidation Uno C-stat for predictions from all fold models: {:.3f}\n".format(val_all_unoc))
                out_file.write('\n')
                out_file.write("Performance of best fold, {}:\n".format(best_fold))
                out_file.write("Best Training Loss: {:.3f} \tBest Validation Loss: {:.3f} \n"
                            "Best Training C-index: {:.3f} \tBest Validation C-index: {:.3f} \n"
                            "Best Training Uno C-stat: {:.3f} \tBest Validation Uno C-stat: {:.3f} \n".format(
                                tr_final_loss, val_final_loss, tr_final_cind, val_final_cind, tr_final_unoc, val_final_unoc))
                out_file.write('\n')



    # Return model, loss, and c-ind from the best fold
    return best_model, best_fold_train_preds, best_fold_valid_preds, tr_final_loss, tr_final_cind, tr_final_unoc, val_final_loss, val_final_cind, val_final_unoc


def test_model(model, test_data_info_path, test_data_img_path, train_data_info_path, train_data_img_path, device, train_predictions, conf_int_count=5):
    """
    Function to test a pytorch model

    Args:
        model: pytorch model (nn.Module), trained model to run testing data through
        data_info_path: str, file path to label spreadsheet for testing data
        data_img_path: str, file path to directory containing testing image data
        device: torch.device, device to use for testing (GPU or CPU)

    Returns:
        test_loss: float, criterion result for model predictions on testing set, calculated per batch and averaged
        test_cind: float, concordance index for model predictions on testing set, calculated over entire set
        df_all_pred: pandas Dataframe, table of each slice's prediction, time and event labels
    """
    # Create Dataset object for test data
    test_dataset = HDFSTumorDataset(test_data_info_path, test_data_img_path, args.ORIG_IMG_DIM, args.TRANSFORM_LIST)
    # Set up DataLoader for test data
    test_loader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE, shuffle=True, drop_last=False)

    # Create Dataset object for train data
    train_dataset = HDFSTumorDataset(train_data_info_path, train_data_img_path, args.ORIG_IMG_DIM, args.TRANSFORM_LIST)
    train_loader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=True, drop_last=False)

    # Put model on GPU if available
    model.to(device)


    # Setting loss function
    criterion = NegativeLogLikelihood(device)

    test_loss, test_cind, test_unoc, test_predictions = valid_epoch(model, device, test_loader, criterion, train_loader)

    # Print evaluation metrics
    print("Testing Results:")
    print("Testing loss: {:.3f}\tTesting c-index: {:.3f}\tTesting Uno C-stat: {:.3f}".format(test_loss, test_cind, test_unoc))

    # Confidence intervals
    test_info = pd.read_csv(test_data_info_path)
    num_samples = test_info.shape[0]
    df_unoc_confidence = pd.DataFrame(columns=['UnoC'])
    np_train_time = np.array(train_predictions['Time'], dtype=float)
    np_train_event = np.array(train_predictions['Event'], dtype=float)

    for x in range(conf_int_count):
        # Set up random sampler
        samp_test_dataset = RandomSampler(test_dataset, replacement=True, num_samples=num_samples)

        samp_loader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE, sampler=samp_test_dataset, drop_last=False)

        samp_loss, samp_all_cind, _, samp_predictions = valid_epoch(model, device, samp_loader, criterion, train_loader)

        np_samp_time = np.array(samp_predictions['Time'], dtype=float)
        np_samp_event = np.array(samp_predictions['Event'], dtype=float)
        np_samp_predictions = np.array(samp_predictions['Prediction'], dtype=float)

        samp_test_cstat = uno_c_statistic(np_train_time, np_train_event, np_samp_time, np_samp_event, np_samp_predictions)

        df_unoc_confidence.loc[x] = samp_test_cstat

    return test_loss, test_cind, test_unoc, test_predictions, df_unoc_confidence


# Confidence check moved to test_model
# def confidence_check(train_predictions, test_info_path, test_img_path, model, device):

#     model.to(device)
#     criterion = NegativeLogLikelihood(device)

#     # Load test label file to get number of test patients
#     test_info = pd.read_csv(test_info_path)
#     num_samples = test_info.shape[0]

#     # Create Dataset object for test data
#     test_dataset = HDFSTumorDataset(test_info_path, test_img_path, args.ORIG_IMG_DIM, args.TRANSFORM_LIST)

#     df_unoc_confidence = pd.DataFrame(columns=['UnoC'])

#     np_train_time = np.array(train_predictions['Time'], dtype=float)
#     np_train_event = np.array(train_predictions['Event'], dtype=float)

#     for x in range(5):
#         # Setting up random sampler
#         samp_test_dataset = RandomSampler(test_dataset, replacement=True, num_samples=num_samples)

#         # Set up DataLoader for test data
#         test_loader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE, sampler=samp_test_dataset, drop_last=False)

#         test_loss, test_all_cind, _, test_predictions = valid_epoch(model, device, test_loader, criterion)

#         np_test_time = np.array(test_predictions['Time'], dtype=float)
#         np_test_event = np.array(test_predictions['Event'], dtype=float)
#         np_test_predictions = np.array(test_predictions['Prediction'], dtype=float)

#         samp_test_cstat = uno_c_statistic(np_train_time, np_train_event, np_test_time, np_test_event, np_test_predictions)

#         df_unoc_confidence.loc[x] = samp_test_cstat

#     return df_unoc_confidence


def train_main():
    """
    Main function to run if training a model from scratch
    """
    # Output setup
    out_dir = 'Output/' + args.MODEL_NAME + '/' + args.IMAGE_TYPE + '/' + datetime.now().strftime("%Y_%m_%d_%H%M") + "_train"
    out_path = os.path.join(args.DATA_DIR, out_dir)

    if not args.DEBUG:
        # Make output directory to save results to
        os.makedirs(out_path)

        # Save out parameters used for the run
        save_param_fname = os.path.join(out_path, 'parameters.txt')
        with open(save_param_fname, 'w') as f:
            with contextlib.redirect_stdout(f):
                help(args)

    # Set up loading paths for label data
    train_info_path = os.path.join(args.DATA_DIR, args.TRAIN_LABEL_FILE)
    test_info_path = os.path.join(args.DATA_DIR, args.TEST_LABEL_FILE)

    # Set up loading ptahs for image data
    train_img_path = os.path.join(args.DATA_DIR, args.IMG_LOC_PATH, 'train/')
    test_img_path = os.path.join(args.DATA_DIR, args.IMG_LOC_PATH, 'test/')

    # Train model with k-fold cross-validation
    best_model, train_predictions, valid_predictions, train_loss, train_cind, train_cpe, valid_loss, valid_cind, valid_cpe = kfold_train(train_info_path, train_img_path, out_path, k=args.K)
    # torch.cuda.empty_cache()

    # Run the trained model through testing
    test_loss, test_cind, test_unoc, test_predictions, df_unoc_confidence = test_model(best_model, test_info_path, test_img_path, train_info_path, train_img_path, device, train_predictions, conf_int_count=1000)

    # Output model shape/setup for reference
    model_stats = summary(best_model, input_size=(args.BATCH_SIZE, 1, args.ORIG_IMG_DIM, args.ORIG_IMG_DIM))
    summary_str = str(model_stats)

    if not args.DEBUG:
        # Add testing results to results.txt file made in kfold_train
        with open(os.path.join(out_path, 'k_fold_results.txt'), 'a') as kfold_file:
            kfold_file.write("Testing Results of Best Fold: \n")
            kfold_file.write("Testing Loss: {:.3f}\tTesting C-Index: {:.3f}\tTesting Uno C-stat: {:.3f}\n\n".format(test_loss, test_cind, test_unoc))

        # Save summary of model
        with open(os.path.join(out_path, 'model_summary.txt'), 'w') as out_file:
            out_file.write(summary_str)

        # Save best model
        model_file_name = 'k_cross_' + args.MODEL_NAME + '.pt'
        torch.save(best_model, os.path.join(out_path, model_file_name))

        # Saving predictions made for all data sets
        train_predictions.to_csv(os.path.join(out_path, 'train_predictions.csv'), index=False)
        valid_predictions.to_csv(os.path.join(out_path, 'valid_predictions.csv'), index=False)
        test_predictions.to_csv(os.path.join(out_path, 'test_predictions.csv'), index=False)

        # Save confidence interval runs
        df_unoc_confidence.to_csv(os.path.join(out_path, 'CNN_unoC_confidence_intervals.csv'), index=False)

    print("Completed model training/testing")

    if args.CONFIDENCE_CHECK == True:
        df_unoc_confidence = confidence_check(train_predictions, test_info_path, test_img_path, best_model, device)

        if not args.DEBUG:
            df_unoc_confidence.to_csv(os.path.join(out_path, 'CNN_unoC_confidence.csv'), index=False)



def load_main():
    """
    Main function if loading a pre-trained model to just run testing on
    """
    # Output setup
    out_dir = 'Output/' + args.MODEL_NAME + '/' + args.IMAGE_TYPE + '/' + datetime.now().strftime("%Y_%m_%d_%H%M") + "_load"
    out_path = os.path.join(args.DATA_DIR, out_dir)

    if not args.DEBUG:
        # Make output directory to save results in
        os.makedirs(out_path)

        # Save out parameters used for the run
        save_param_fname = os.path.join(out_path, 'parameters.txt')
        with open(save_param_fname, 'w') as f:
            with contextlib.redirect_stdout(f):
                help(args)

    # Set up loading paths for testing labels and image data
    test_info_path = os.path.join(args.DATA_DIR, args.TEST_LABEL_FILE)
    test_img_path = os.path.join(args.DATA_DIR, args.IMG_LOC_PATH, 'test/')

    # Set up loading paths for training labels and image data
    train_info_path = os.path.join(args.DATA_DIR, args.TRAIN_LABEL_FILE)
    train_img_path = os.path.join(args.DATA_DIR, args.IMG_LOC_PATH, 'train/')


    # Load the pre-trained model
    load_model = torch.load(args.LOAD_MODEL_PATH)

    # Run testing data through the pre-trained model
    test_loss, test_cind, test_unoc, test_predictions = test_model(load_model, test_info_path, test_img_path, train_info_path, train_img_path, device)

    if not args.DEBUG:
        # Save testing results
        with open(os.path.join(out_path, 'k_fold_results.txt'), 'a') as kfold_file:
            kfold_file.write("Testing Results of Loaded Model: \n")
            kfold_file.write("Testing Loss: {:.3f} \t Testing C-Index: {:.2f}\n\n".format(test_loss, test_cind))

        # Saving predictions made for test data
        test_predictions.to_csv(os.path.join(out_path, 'test_predictions.csv'), index=False)

    print("Completed loaded model testing")


if __name__ == '__main__':
    # Preliminaries
    # torch.cuda.empty_cache()
    # Use GPU if available
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    # Random setup - setting random seeds for reproducibility
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
