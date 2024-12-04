import torch
import copy
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
from algorithms.fedpsd.ClientTrainer import ClientTrainer
from algorithms.BaseServer import BaseServer

__all__ = ["Server"]


class Server(BaseServer):
    def __init__(
            self, algo_params, model, data_distributed, optimizer, scheduler, **kwargs
    ):
        super(Server, self).__init__(
            algo_params, model, data_distributed, optimizer, scheduler, **kwargs
        )
        self.client = ClientTrainer(
            algo_params=self.algo_params,
            model=copy.deepcopy(model),
            local_epochs=self.local_epochs,
            device=self.device,
            num_classes=self.num_classes,
        )


        print("\n>>> Distill-Fedavg Server initialized...\n")

    def run(self):
        self._print_start()

        for round_idx in range(self.n_rounds):

            if round_idx == 0:
                test_acc = evaluate_model(self.model, self.testloader, device=self.device)
                self.server_results["test_accuracy"].append(test_acc)

            self.client.round_alpha = (round_idx + 1) / self.n_rounds

            start_time = time.time()

            sampled_clients = self._client_sampling(round_idx)


            self.server_results["client_history"].append(sampled_clients)

            updated_local_weights, client_sizes, round_results = self._clients_training(sampled_clients)

            ag_weights = self._aggregation(updated_local_weights, client_sizes)

            self._update_and_evaluate(ag_weights, round_results, round_idx, start_time)

    def _clients_training(self, sampled_clients):


        updated_local_weights, client_sizes = [], []
        round_results = {}


        server_weights = self.model.state_dict()
        server_optimizer = self.optimizer.state_dict()


        for client_idx in sampled_clients:

            self.client_round_counts += 1
            self.client.current_client_round = self.client_round_counts

            self._set_client_data(client_idx)
            self.client.download_global(server_weights, server_optimizer)

            self.client.last_output = self.all_client_last_output

            local_results, local_size, client_last_output = self.client.train()

            self.all_client_last_output = client_last_output


            updated_local_weights.append(self.client.upload_local())

            round_results = self._results_updater(round_results, local_results)
            client_sizes.append(local_size)

            self.client.reset()


        return updated_local_weights, client_sizes, round_results

    def _set_client_data(self, client_idx):

        self.client.datasize = self.data_distributed["local"][client_idx]["datasize"]
        self.client.trainloader = self.data_distributed["local"][client_idx]["train"] 
        self.client.testloader = self.data_distributed["global"]["test"]  
        self.client.data_distribution_map = self.data_distributed["data_map"][client_idx]