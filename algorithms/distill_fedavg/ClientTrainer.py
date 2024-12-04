import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.BaseClientTrainer import BaseClientTrainer

__all__ = ["ClientTrainer"]


class ClientTrainer(BaseClientTrainer):
    def __init__(self, **kwargs):
        super(ClientTrainer, self).__init__(**kwargs)
        self.round_alpha = None
        self.data_distribution_map = None
    def train(self):
        """Local training"""
        self.model.train()
        self.model.to(self.device)
        local_size = self.datasize
        if self.current_client_round == 1:
            for _ in range(self.local_epochs):
                for (data, targets, input_indices) in self.trainloader:
                    self.optimizer.zero_grad()
                    # forward pass
                    data, targets = data.to(self.device), targets.to(self.device)
                    output = self.model(data)
                    
                    adapt_output = self.calibration_logits(output, self.data_distribution_map)
                    softmax_output = F.softmax(output, dim=1).detach()

                    loss = F.nll_loss(adapt_output, targets)
                    loss.backward()
                    self.optimizer.step()

                    for i, key in enumerate(input_indices):
                        self.last_output[key.item()] = softmax_output[i]

        else:
            for _ in range(self.local_epochs):
                for (data, targets, input_indices) in self.trainloader:
                    self.optimizer.zero_grad()
                    # forward pass
                    data, targets = data.to(self.device), targets.to(self.device)
                    output = self.model(data)
                    adapt_output = self.calibration_logits(output, self.data_distribution_map)
                    softmax_output = F.softmax(output, dim=1).detach()

                    last_output = torch.tensor([list(self.last_output[key.item()]) for key in input_indices]).to(
                        self.device)
                    targets_one_hot = F.one_hot(targets, self.num_classes).float()

                    soft_targets = self.round_alpha * last_output + (1 - self.round_alpha) * targets_one_hot
                    soft_targets = soft_targets.cuda()

                    adapt_log_softmax = self.calibration_logits(output, self.data_distribution_map)
                    loss_part1 = F.nll_loss(adapt_log_softmax, targets)
                    
                    log_pred_student = F.log_softmax(output, dim=1)
                    pred_teacher = soft_targets  
                    loss_part2 = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
                    
                    loss = loss_part1 + loss_part2
                    
                    loss.backward()
                    self.optimizer.step()

                    for i, key in enumerate(input_indices):
                        self.last_output[key.item()] = softmax_output[i]

        last_output = self.last_output
        local_results = self._get_local_stats()

        return local_results, local_size, last_output


