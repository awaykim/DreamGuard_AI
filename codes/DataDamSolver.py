import os
import numpy as np
import torch
from sklearn.metrics import roc_curve, roc_auc_score, fbeta_score,confusion_matrix
from sklearn.preprocessing import label_binarize
import logging


class DataDamSolver(object):
    def __init__(self, config):
        self.config = config

        self.model = config["model"]
        self.optimizer = config["optimizer"]
        self.loss_function = config["loss_function"]
        self.label_name = 'hr_sleep'
        self.acc = 0
        self.auroc=0
        self.cutoff = 0

        #logging.basicConfig(filename=os.path.join(self.config['save_path'],"example.log"), filemode='w',level=logging.INFO)

        self.logger = logging.getLogger('myApplication')
        self.logger.setLevel(logging.DEBUG)

        # create file handler which logs even debug messages
        self.fh = logging.FileHandler(os.path.join(self.config['save_path'],"example.log"))
        fileLogLevel=logging.INFO 
        self.fh.setLevel(fileLogLevel)

        self.logger.addHandler(self.fh)

    def logging(self,log):
        print(log)
        self.logger.info(log)

    def fit(self):
        for epoch in range(self.config["epochs"]):
            loss = self._training(
                loader=self.config["dataloader"]["train"], epoch=str(epoch)
            )
            y,y_prob, (auroc, acc, f_beta,sens,spec,cutoff,confusion) = self._test(loader=self.config["dataloader"]["valid"], epoch=str(epoch))

            
            if acc>self.acc:
                print(f'##### {epoch} MODEL SAVED #####')
                self.logger.info(f'##### {epoch} MODEL SAVED #####')
                torch.save(
                    {"config": self.config, "weight": self.model.state_dict()},
                    os.path.join(self.config["save_path"], f"checkpoint_{self.label_name}.pth"),
                )
                self.acc = acc
                self.auroc = auroc
                self.cutoff = cutoff
            
            if acc==self.acc:
                if auroc>self.auroc:
                    print(f'##### {epoch} MODEL SAVED #####')
                    self.logger.info(f'##### {epoch} MODEL SAVED #####')
                    torch.save(
                        {"config": self.config, "weight": self.model.state_dict()},
                        os.path.join(self.config["save_path"], f"checkpoint_{self.label_name}.pth"),
                    )
                    self.acc = acc
                    self.auroc = auroc
                    self.cutoff = cutoff


        self._test(loader=self.config["dataloader"]["test"], epoch="test",cutoff=self.cutoff)

        self.fh.close()
        self.logger.removeHandler(self.fh)    


        
    def feed_forward(self, ecg, y):
        ecg = ecg.cuda()
        y = y.cuda().long()
        prob = self.model(ecg)
        loss = self.loss_function(prob, y)

        return y, prob, loss

    def _training(self, loader, epoch):
        self.model.train()

        for _, (_, x, y) in enumerate(loader):
            y, prob, loss = self.feed_forward(x, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print(f"epoch {epoch} => loss: {loss}")

        return loss

    def _test(self, loader, epoch,cutoff=None):
        y_label, y_prob = [], []
        
        if epoch =='test':
            self.model.load_state_dict(torch.load(os.path.join(self.config["save_path"], f"checkpoint_{self.label_name}.pth"))['weight'])
        
        self.model.eval()
        with torch.no_grad():
            for i, (fname, x, y) in enumerate(loader):
                y, prob, loss = self.feed_forward(x, y)
                y_label.append(y.detach().cpu()) if y is not None else None
                y_prob.append(prob.detach().cpu()) if prob is not None else None
                loss = 0 if torch.isnan(loss) else loss.detach().cpu()

        auroc, acc, f_beta,sens,spec,cutoff,confusion = get_performance(y_label, y_prob, 2,cutoff)
        print(
            f"epoch {epoch} performance => AUROC: {auroc:.4f}, ACC: {acc:.4f}, F1_score:{f_beta:.4f}, sens:{sens:.4f}, spec:{spec:.4f} confusion:{confusion}"
        )
        self.logger.info(f"epoch {epoch} performance => AUROC: {auroc:.4f}, ACC: {acc:.4f}, F1_score:{f_beta:.4f} sens:{sens:.4f} spec:{spec:.4f} confusion:{confusion}")
        return y_label, y_prob,( auroc, acc, f_beta,sens,spec,cutoff,confusion)


def get_performance(y_true, y_prob, num_classes,cutoff):

    if type(y_true)==list:
        y_true = np.concatenate(y_true, 0)
    if type(y_prob)==list:
        y_prob = np.concatenate(y_prob, 0)

    if num_classes > 2:
        y_true_onehot = label_binarize(y_true, classes=range(num_classes))
        multi_class = "ovo"
    else:
        multi_class = None
        y_true_onehot = y_true
        if num_classes == 2:
            y_prob = y_prob[:, 1:]

    auroc = roc_auc_score(
        y_true_onehot, y_prob, multi_class=multi_class, average="weighted"
    )
    
    if num_classes <= 2:
        fpr, tpr, threshold_auroc = roc_curve(y_true, y_prob)
        if cutoff is None:
            cutoff = threshold_auroc[np.argmax(tpr - fpr)]
        y_pred = (y_prob > np.array(cutoff)) * 1
    else:
        y_pred = np.argmax(y_prob, -1)

    f_beta = fbeta_score(y_true, y_pred, beta=1, average="weighted")

    if y_prob.ndim!=1:
        y_pred = y_prob[:, 0]
    y_pred[y_pred <cutoff] = 0
    y_pred[y_pred >= cutoff] = 1
    acc = sum(np.equal(y_pred, y_true)) / len(y_true)
    acc = sum(np.equal(y_pred, y_true)) / len(y_true)
    (TN, FP, FN, TP) = confusion_matrix(y_true,y_pred,labels=[0,1]).ravel()
    
    try:
        sens = TP / (TP + FN)
    except ZeroDivisionError:
        sens = None
    
    try:
        spec = TN / (TN + FP)
    except ZeroDivisionError:
        spec = None


    return auroc, acc, f_beta,sens,spec,cutoff, (TN, FP, FN, TP)
