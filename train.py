
import os
import argparse
from utils import *
from torch.utils.data import Dataset
from pysurvival.models.simulations import SimulationModel
from models import *


parser = argparse.ArgumentParser(description='EPL')
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--dropout', default=.3, type=float)
parser.add_argument('--out', default=1, type=int)
parser.add_argument('--lib', default='', type=str)
parser.add_argument('-e', '--epochs', default=2000, type=int)
parser.add_argument('--decay-interval', default=400, type=int)
parser.add_argument('-b','--batch-size', default=4000, type=int)
parser.add_argument('--weightdecay', default=1e-4, type=float)
parser.add_argument('--covariates', default=10, type=int)
parser.add_argument('--strat', default='none', type=str)
parser.add_argument('--development', default=0, type=int)
parser.add_argument('--activation', default='SELU', type=str)

def main():
    global args, gpu, best_acc
    best_acc = 0
    gpu = torch.device("cpu")
    args = parser.parse_args()
    root_output = '/Users/katyscott/Documents/ICC/Code/cox_experiments'
    if args.development == 1:
        save_path = 'test'
    else:
        save_path = '{}_{}lr_{}b_'.format(args.activation,args.lr,args.batch_size)
    out_dir = os.path.join(root_output, save_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    model = BasicModel(args.activation).to(gpu)

    # picking loss as normal neg log likelihood or with extra stratification boosting parameter
    if args.strat == 'none':
        criterion = NegativeLogLikelihood(gpu)
    else:
        criterion = NegativeLogLikelihoodStrat(gpu)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    sim = SimulationModel(survival_distribution='exponential',risk_type='Linear',censored_parameter=6,alpha=1, beta=5) # generate random survival times with exp. distribution
    train_samples = sim.generate_data(num_samples=4000, num_features=args.covariates,
                                      feature_weights = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    val_samples = sim.generate_data(num_samples=500, num_features=args.covariates,
                                    feature_weights = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0])

    train_dataset = SurvivalDataset(train_samples, args)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_dataset.__len__())
    val_dataset = SurvivalDataset(val_samples, args)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_dataset.__len__())

    for epoch in range(0,args.epochs):
        coxLossMeter = AverageMeter()
        stratLossMeter = AverageMeter()
        ciMeter = AverageMeter()
        varMeter = AverageMeter()
        model.train()
        for X, y, e in train_loader:
            risk_pred = model(X.float().to(gpu))
            if args.strat == 'median':
                low, high = stratify(risk_pred)
                cox_loss, strat_loss = criterion(-risk_pred, y.to(gpu), e.to(gpu), low, high)
                train_loss = cox_loss + strat_loss
            else:
                cox_loss = criterion(-risk_pred, y.to(gpu), e.to(gpu), model)
                strat_loss = torch.Tensor([0])
                train_loss = cox_loss
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            coxLossMeter.update(cox_loss.item(), y.size(0))
            stratLossMeter.update(strat_loss.item(), y.size(0))
            varMeter.update(risk_pred.var(), y.size(0))
            train_c = c_index(risk_pred, y, e)
            ciMeter.update(train_c.item(), y.size(0))


        model.eval()
        ciValMeter = AverageMeter()
        for X, y, e in val_loader:
            risk_pred = model(X.float().to(gpu))
            val_c = c_index(risk_pred, y, e)
            ciValMeter.update(val_c.item(), y.size(0))
        print('Epoch: {} \t Train Loss: {:.4f} \t Train CI: {:.3f} \t Val CI: {:.3f}'.format(epoch, train_loss, train_c, val_c))
        save_error(ciMeter.avg, ciValMeter.avg, coxLossMeter.avg, stratLossMeter.avg, varMeter.avg, epoch, os.path.join(out_dir, 'convergence.csv'))



def stratify(risk):
    ordered_risks, order_idx = torch.sort(risk, 0)
    low = risk[order_idx[(ordered_risks < torch.median(ordered_risks))]]
    high = risk[order_idx[(ordered_risks >= torch.median(ordered_risks))]]
    return low, high


if __name__ == '__main__':
    main()
