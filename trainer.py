import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.model import Model
from tqdm import tqdm
import numpy as np
import torch
from datasets.dataset import LoadDataset
from evaluator import Evaluator
from torch.nn import CrossEntropyLoss
from losses.double_alignment import CORAL,CORAL_kendall,CORAL_spearman,CORALWithRBF
from losses.ae_loss import AELoss
from timeit import default_timer as timer
import os
import copy
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import gc
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from sklearn.utils.class_weight import compute_class_weight
import wandb
from focal_loss import FocalLoss
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from collections import defaultdict
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, verbose=False):
        """
        Args:
            patience (int): how many epochs to wait after last improvement.
            min_delta (float): minimum change in monitored value to qualify as improvement.
            verbose (bool): whether to print messages when improvement or stopping occurs.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_acc = None
        self.counter = 0
        self.early_stop = False
        self.best_model_states = None

    def __call__(self, val_acc, model):
        if self.best_acc is None:
            self.best_acc = val_acc
            self.best_model_states = copy.deepcopy(model.state_dict())
            if self.verbose:
                print(f"[EarlyStopping] Initializing best_acc = {val_acc:.4f}")
        elif val_acc > self.best_acc + self.min_delta:
            self.best_acc = val_acc
            self.best_model_states = copy.deepcopy(model.state_dict())
            self.counter = 0
            if self.verbose:
                print(f"[EarlyStopping] Improvement detected: val_acc = {val_acc:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] No improvement for {self.counter}/{self.patience} epochs.")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"[EarlyStopping] Triggered early stopping. Best acc = {self.best_acc:.4f}")


class Trainer(object):
    def __init__(self, params):
        self.params = params
        self.data_loader, subject_id = LoadDataset(params).get_data_loader()
        self.target_dataset=self.params.target_domains
        print(self.target_dataset)
        self.val_eval = Evaluator(params, self.data_loader['val'])
        self.test_eval = Evaluator(params, self.data_loader['test'])

        wandb.init(
            project=self.params.project_name,  
            name=f"{self.params.target_domains}_{self.params.run_name}",  
            config=self.params.__dict__  
        )
        
        self.best_model_states = None

        self.model = Model(params).cuda()
        #self.ce_loss = CrossEntropyLoss(label_smoothing=self.params.label_smoothing).cuda() # classification loss
        #alpha=[0.8732, 2.0817, 0.5244, 1.2010, 1.5753] #calculated class weight from training data
        alpha=[1] * self.params.num_of_classes
        if self.params.use_focal:
            self.focal_loss=FocalLoss(gamma=0.1,alpha=alpha,reduction='mean',task_type='multi-class',num_classes=self.params.num_of_classes).cuda()
        else:
            self.ce_loss = CrossEntropyLoss(label_smoothing=self.params.label_smoothing).cuda() # classification loss
        self.coral_loss = CORAL().cuda()
        self.ae_loss = AELoss().cuda() # reconstruction loss
        if self.params.optimizer == "RAdam":
            self.optimizer = torch.optim.RAdam(
                self.model.parameters(),
                lr=self.params.lr,                    
                betas=(0.9, 0.999),         
                eps=1e-8,                  
                weight_decay=self.params.lr/10,)            
            print("RAdam")
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.lr, weight_decay=self.params.lr/10)
            print("Adam")

        self.data_length = len(self.data_loader['train'])    
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.params.epochs * self.data_length
        )

        #print(self.model)
        self.early_stopper = EarlyStopping(
                patience=self.params.patience,      
                min_delta=self.params.min_delta,    
                verbose=True
            )


    def train(self):
        acc_best = 0
        f1_best = 0
        i = 0
        accumulation_steps = self.params.accumulation_steps
        for epoch in range(self.params.epochs):
            #print(epoch)
            loss1_list, loss2_list, loss3_list = [], [], []
            loss2a_list,loss2b_list,loss2c_list=[],[],[]
            loss1a_list,loss1b_list=[],[]
            global_step = 0
            self.model.train()
            start_time = timer()
            losses = []
            self.optimizer.zero_grad()

            for batch_idx,(x, y, z) in enumerate(tqdm(self.data_loader['train'], mininterval=10)):
                #self.optimizer.zero_grad()
                x = x.cuda()
                #print(x.shape) 

                y = y.cuda()
                z = z.cuda()
                #print(z.shape)
                if self.params.return_attention:
                    pred, recon, mu,attn_weights = self.model(x)

                else:
                    pred, recon, mu = self.model(x)
                
                if not self.params.use_focal:
                    loss1 = self.ce_loss(pred.transpose(1, 2), y) #for cross entropy loss
                else:
                    loss1=self.focal_loss(pred,y)
                #print(loss1)
                (loss2_a,loss2_b),loss2_c = self.coral_loss(mu, z)
                loss3 = self.ae_loss(x, recon) 
  
                loss = loss1 + (0.5*(loss2_a + loss2_b + loss2_c))  + (0.5*loss3)
                loss = loss / accumulation_steps
                loss.backward()
                losses.append(loss.data.cpu().numpy() * accumulation_steps) 
                loss1_list.append(loss1.item())
                loss2a_list.append(loss2_a.item())
                loss2b_list.append(loss2_b.item())
                loss2c_list.append(loss2_c.item())
                loss3_list.append(loss3.item())
                wandb.log({
                    "batch/loss1": loss1.item(),
                    "batch/loss2_a": loss2_a.item(),
                    "batch/loss2_b": loss2_b.item(),
                    "batch/loss2_c": loss2_c.item(),
                    "batch/loss3": loss3.item(),
                    "batch/total_loss": (loss1 +  0.5 * (loss2_a +  loss2_b + loss2_c) + 0.5 * loss3).item(),
                    "step": global_step
                })
                global_step += 1
 
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)

                if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(self.data_loader['train'])):
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
            optim_state = self.optimizer.state_dict()
            with torch.no_grad():
                acc, f1, cm, wake_f1, n1_f1, n2_f1, n3_f1, rem_f1 = self.val_eval.get_accuracy(self.model)
                #print(losses)
                print(
                    "Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, f1: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        acc,
                        f1,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                print(cm)
                print(
                    "wake_f1: {:.5f}, n1_f1: {:.5f}, n2_f1: {:.5f}, n3_f1: {:.5f}, rem_f1: {:.5f}".format(
                        wake_f1,
                        n1_f1,
                        n2_f1,
                        n3_f1,
                        rem_f1,
                    )
                )
                avg_loss1 = np.mean(loss1_list)
                avg_loss2_a = np.mean(loss2a_list)
                avg_loss2_b = np.mean(loss2b_list)
                avg_loss2_c = np.mean(loss2c_list)
                avg_loss3 = np.mean(loss3_list)
                avg_total_loss = avg_loss1 + 0.5 * (avg_loss2_a + avg_loss2_b +  avg_loss2_c) + 0.5 * avg_loss3
                wandb.log({
                    'epoch': epoch + 1,
                    'loss1': avg_loss1,
                    'loss2_a': avg_loss2_a,
                    'loss2_b': avg_loss2_b,
                    'loss2_c': avg_loss2_c,
                    'loss3': avg_loss3,
                    'train_total_loss': avg_total_loss,
                    'val_acc': acc,
                    'val_f1': f1,
                    'val_f1_wake': wake_f1,
                    'val_f1_n1': n1_f1,
                    'val_f1_n2': n2_f1,
                    'val_f1_n3': n3_f1,
                    'val_f1_rem': rem_f1,
                    'learning_rate': optim_state['param_groups'][0]['lr'],
                })

                if self.early_stopper.early_stop:
                    print(f"Stopping early at epoch {epoch+1}")
                    break
                if acc > acc_best:
                    best_f1_epoch = epoch + 1
                    acc_best = acc
                    f1_best = f1
                    self.best_model_states = copy.deepcopy(self.model.state_dict())
                    print("Epoch {}: ACC increasing!! New acc: {:.5f}, f1: {:.5f}".format(best_f1_epoch, acc_best, f1_best))
                # if f1 > f1_best:
                #     best_f1_epoch = epoch + 1
                #     acc_best = acc
                #     f1_best = f1
                #     self.best_model_states = copy.deepcopy(self.model.state_dict())
                #     print("Epoch {}: F1 increasing!! New acc: {:.5f}, f1: {:.5f}".format(best_f1_epoch, acc_best, f1_best))
        print("{} epoch get the best acc {:.5f} and f1 {:.5f}".format(best_f1_epoch, acc_best, f1_best))
        test_acc, test_f1 = self.test()
        return test_acc, test_f1

    def test(self):
        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            test_acc, test_f1, test_cm, test_wake_f1, test_n1_f1, test_n2_f1, \
            test_n3_f1, test_rem_f1 = self.test_eval.get_accuracy(self.model)
            print("***************************Test results************************")
            print(
                "Test Evaluation: acc: {:.5f}, f1: {:.5f}".format(
                    test_acc,
                    test_f1,
                )
            )
            print(test_cm)
            print(
                "wake_f1: {:.5f}, n1_f1: {:.5f}, n2_f1: {:.5f}, n3_f1: {:.5f}, rem_f1: {:.5f}".format(
                    test_wake_f1,
                    test_n1_f1,
                    test_n2_f1,
                    test_n3_f1,
                    test_rem_f1,
                )
            )
            wandb.log({
                'test_acc': test_acc,
                'test_f1': test_f1,
                'test_f1_wake': test_wake_f1,
                'test_f1_n1': test_n1_f1,
                'test_f1_n2': test_n2_f1,
                'test_f1_n3': test_n3_f1,
                'test_f1_rem': test_rem_f1,
            })
  
            log_file=self.params.log_file
            logging.basicConfig(
                filename=log_file,
                filemode='a', 
                format='%(asctime)s - %(levelname)s - %(message)s',
                level=logging.INFO
            )
            logging.info(f"Target dataset: {self.target_dataset}")
            logging.info(f"Test Evaluation: acc: {test_acc}, f1: {test_f1}")
            logging.info(f"wake_f1: {test_wake_f1}, n1_f1: {test_n1_f1}, n2_f1: {test_n2_f1}, n3_f1: {test_n3_f1}, rem_f1: {test_rem_f1}")

            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir,exist_ok=True)
            model_path = self.params.model_dir +"/{}_tacc_{:.5f}_tf1_{:.5f}.pth".format(
                self.target_dataset,
                test_acc,
                test_f1,
            )
            torch.save(self.best_model_states, model_path)
            print("the model is save in " + model_path)
        return test_acc, test_f1
