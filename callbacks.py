import pandas as pd
import torch
import numpy as np
from funcs import getMetrics,DataCreation,vison_utils
from utils.visualizer import Visualizer
import os,cv2

from catalyst.dl  import  Callback, CallbackOrder,Runner

class MetricsCallback(Callback):

    def __init__(self,
                 directory=None,
                 model_name='',
                 check_interval=10,
                 input_key: str = "targets",
                 output_key: str = "logits",
                 prefix: str = "acc_pre_rec_f1",

                 ):
        global visualizer,count
        super().__init__(CallbackOrder.Metric)

        self.input_key = input_key
        self.output_key = output_key
        self.prefix = prefix
        self.directory = directory
        self.model_name = model_name
        self.check_interval = check_interval
        if 'visualizer' not in globals():
            visualizer = Visualizer()
            count=0
        self.visualizer = visualizer




    def on_epoch_end(self, state: Runner):
        """Event handler for epoch end.
        Args:
            runner ("IRunner"): IRunner instance.
        """
        #print(torch.sum(state.model.fc1.weight),state.model.fc1.weight[5][300])
        #print(torch.sum(state.model.conv_blocks[0].conv1.weight))
        global count
        if count % self.check_interval == 0:
            if self.directory is not None: torch.save(state.model.state_dict(), str(self.directory) + '/' +
                                                  self.model_name + "_" + str(
            state.stage_epoch_step) + ".pth")

            preds = torch.argmax(state.batch['logits'], dim=1)
            #print("{} is {}".format(self.prefix, getMetrics(state.batch['targets'], preds)))
        self.visualizer.display_current_results(state.stage_epoch_step+count, state.epoch_metrics['train']['loss'],
                                                name='train_loss')

        count+=1
