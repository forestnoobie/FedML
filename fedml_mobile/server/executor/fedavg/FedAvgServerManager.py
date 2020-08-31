import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../")))

import logging

from fedml_core.distributed.communication.message import Message
from fedml_api.distributed.fedavg.message_define import MyMessage
from fedml_core.distributed.server.server_manager import ServerManager


class FedAVGServerManager(ServerManager):
    def __init__(self, args, aggregator):
        super().__init__(args, backend="MQTT")
        self.args = args
        self.aggregator = aggregator
        self.round_num = args.comm_round
        self.round_idx = 0

    def run(self):
        global_model_params = self.aggregator.get_global_model_params()
        for process_id in range(1, self.size):
            self.send_message_init_config(process_id, global_model_params)
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
                                              self.handle_message_receive_model_from_client)

    def handle_message_receive_model_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)
        train_acc = msg_params.get(MyMessage.MSG_ARG_KEY_LOCAL_TRAINING_ACC)
        train_loss = msg_params.get(MyMessage.MSG_ARG_KEY_LOCAL_TRAINING_LOSS)

        self.aggregator.add_local_trained_result(sender_id - 1, model_params, local_sample_number, train_acc,
                                                 train_loss)
        b_all_received = self.aggregator.check_whether_all_receive()
        logging.info("b_all_received = " + str(b_all_received))
        if b_all_received:
            global_model_params = self.aggregator.aggregate()
            self.aggregator.infer(self.round_idx)
            self.aggregator.statistics(self.round_idx)

            # start the next round
            self.round_idx += 1
            if self.round_idx == self.round_num:
                self.finish()
                return

            # since we use MQTT, every client can observe this message, so there is no need to send one by one
            # for receiver_id in range(1, self.args.client_number+1):
            self.send_message_sync_model_to_client(0, global_model_params)

    def send_init_msg(self):
        global_model_params = self.aggregator.get_global_model_params()
        for process_id in range(1, self.size):
            self.send_message_init_config(process_id, global_model_params)

    def send_message_init_config(self, receive_id, global_model_params):
        message = Message(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        self.send_message(message)

    def send_message_sync_model_to_client(self, receive_id, global_model_params):
        logging.info("send_message_sync_model_to_client")
        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        self.send_message(message)