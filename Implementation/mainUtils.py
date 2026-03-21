import torch
import subprocess
import logging
import time
import numpy as np
from scapy.all import rdpcap, TCP
from tqdm import tqdm
from train_evalCopy import train
import argparse
import pickle
import PEAN_model_copy
import os
import pyshark

logger = logging.getLogger(__name__)

def get_k_fold_data(k, i, X):
    assert k > 1
    fold_size = len(X) // k

    X_train = None
    for j in range(k):
        X_part = X[j * fold_size: (j + 1) * fold_size]
        if j == i:
            X_valid = X_part
        elif X_train is None:
            X_train = X_part
        else:
            X_train = X_train + X_part
    return X_train, X_valid


class Config(object):
    def __init__(self,):
        self.model_name = "PEAN"
        pretrain_path = './Model/pretrain/'
        record_path = './Model/record/'
        log_path = './Model/log/'
        loss_path = './Model/loss/'
        save_path = './Model/save/'
        dirs = [pretrain_path, record_path, log_path, loss_path, save_path]
        for dir in dirs:
            if not os.path.exists(dir):
                os.makedirs(dir)

        self.pretrainModel_json = pretrain_path + 'model_128d_8h_2l/config.json'
        self.pretrainModel_path = pretrain_path + 'model_128d_8h_2l/model_128d_8h_2l.pth'
        self.dataset = 'sni_whs'
        self.train_path = './TrafficData/' + '{}_train.txt'.format(self.dataset)
        self.class_list = [x.strip() for x in open('./TrafficData/class.txt').readlines()]
        self.save_path = save_path
        self.record_path = record_path
        self.loss_path = loss_path
        self.log_path = log_path
        self.vocab_path = './Config/vocab.txt'
        self.n_vocab = 261
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dropout = 0.5
        self.require_improvement = 10000
        self.num_classes = len(self.class_list)
        self.bert_dim = 128  # It must be consistent with the settings in pretrain_config.json
        self.num_layers = 2
        self.middle_fc_size = 2048


def setup_main_params():
    parser = argparse.ArgumentParser(description='Traffic Classification')
    
    parser.add_argument('--pcap_folder', default=None, help='Folder from which pcap should be parsed to create dataset')
    parser.add_argument('--pcap_out', default='./TrafficData/sni_whs_train.txt', help='Where to write the extracted pcap files')
    parser.add_argument('--read_pcap', action='store_true', default=False, help='whether to create or not a dataset from pcap files')
    parser.add_argument('--pad_num', type=int, default=10, help='the padding size of packet num')
    parser.add_argument('--pad_len', default=400, type=int, help='the padding size(length) of each packet')
    parser.add_argument('--pad_len_seq', default=10, type=int, help='the padding size of packet length sequence')
    parser.add_argument('--emb', default=128, type=int, help='the emb size of bytes')
    parser.add_argument('--device', default='cuda:0', type=str, help='the training device')
    parser.add_argument('--load', default=True, type=bool, help='whether train on previous model')
    parser.add_argument('--batch', default=64, type=int, help='batch_size')
    parser.add_argument('--feature', default='ensemble', type=str, help='length / raw / ensemble')
    parser.add_argument('--method', default='trf', type=str, help='lstm / trf (Sequential Layer)')
    parser.add_argument('--embway', default='random', type=str, help='random / pretrain (for raw)')
    parser.add_argument('--imploss', default=True, type=bool, help='whether to use improved loss')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--length_emb_size', default=32, type=int, help='len emb size')
    parser.add_argument('--lenhidden', default=128, type=int, help='len hidden size')
    parser.add_argument('--embhidden', default=1024, type=int, help='emb hidden size')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--trf_heads', default=8, type=int, help='transformers heads number')
    parser.add_argument('--trf_layers', default=2, type=int, help='transformers layers')
    parser.add_argument('--mode', default='train', type=str, help='train/test')
    parser.add_argument('--k', default='10', type=int, help='k fold validation')
    parser.add_argument('--epoch', default='300', type=int, help='epoch')
    parser.add_argument('--train', default=True, help='flag to disable training')
    parser.add_argument('--new', default=False, help='flag to disable training')
    args = parser.parse_args()
    if args.new == True:
        args.load = False
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    return args

def get_config(args):
    config = Config()
    config.pad_num = args.pad_num
    config.pad_length = args.pad_len
    config.pad_len_seq = args.pad_len_seq
    config.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')  # 设备
    config.mode = args.mode
    config.embedding_size = args.emb
    config.batch_size = args.batch
    config.load = args.load
    config.lenlstmhidden_size = args.lenhidden
    config.emblstmhidden_size = args.embhidden
    config.feature = args.feature
    config.method = args.method
    config.embway = args.embway
    config.length_emb_size = args.length_emb_size
    config.imploss = args.imploss
    config.learning_rate = args.lr
    config.seed = args.seed
    config.trf_heads = args.trf_heads
    config.trf_layers = args.trf_layers
    config.k = args.k
    config.num_epochs = args.epoch
    if args.mode == "test":
        config.load = True
        config.num_epochs = 0

    name = "{}_{}".format(config.feature, config.seed)
    if config.feature != "length":
        name += "_{}_{}_{}_{}_{}_{}".format(config.embway, config.method, config.embedding_size, config.pad_num,
                                            config.pad_length, config.emblstmhidden_size)
    if config.feature == "length" or config.feature == "ensemble":
        name += "_{}_{}".format(config.pad_len_seq, config.lenlstmhidden_size)
    if config.method == "trf":
        if config.trf_heads == 8 and config.trf_layers == 2:
            pass
        else:
            name += "_{}_{}".format(config.trf_heads, config.trf_layers)
    if config.imploss:
        name += "_imploss"
    config.print_path = config.record_path + name + ".txt"  # record console log
    config.loss_path = config.loss_path + name + ".txt"     # record loss
    config.save_path = config.save_path + name + ".ckpt"    # record saved model
    print("\nModel save at: ", config.save_path)
    from transformers import BertTokenizer
    config.tokenizer = BertTokenizer(vocab_file=config.vocab_path, max_seq_length=config.pad_num - 2, max_len=config.pad_num)

    return config

def prepare_data(config, args):
    print("----------------------------\n")

    with open(config.print_path, 'a') as f:
        f.write("----------------------------\n\n")

    msg = "Iput Feature: {}\nRandom Seed: {}\n".format(config.feature, config.seed)
    if args.feature == "raw" or args.feature == "ensemble":
        msg += "Sequential use: {}\n".format(config.method)
        msg += "Embedding way: {}(hidden:{})\n".format(config.embway, config.emblstmhidden_size)
        if config.method == "pretrain":
            msg += "Bert Size: {}\n".format(config.bert_dim)
        else:
            msg += "Embedding Size: {}\n".format(config.embedding_size)
        msg += "Pad_num: {}\n".format(config.pad_num)
        msg += "Pad_len: {}\n".format(config.pad_length)

    if config.feature == "length" or args.feature == "ensemble":
        msg += "Length use: lstm(emb: {}, hidden:{})\n".format(config.length_emb_size, config.lenlstmhidden_size)
        msg += "Pad_len_seq: {}\n".format(config.pad_len_seq)

    if config.method == "trf":
        msg += "trf heads:{}\n".format(config.trf_heads)
        msg += "trf_layers: {}\n".format(config.trf_layers)

    msg += "Use Improved loss: {}\n".format(config.imploss)
    msg += "Learning Rate: {}\n".format(config.learning_rate)
    msg += "Batch Size:{}\n".format(config.batch_size)
    msg += f"Number of classes: {config.num_classes}\n"

    print(msg)
    with open(config.print_path, 'a') as f:
        f.write(msg)
    print("----------------------------\n")
    with open(config.print_path, 'a') as f:
        f.write("----------------------------\n\n")

    print("Loading data...")
    with open(config.print_path, 'a') as f:
        f.write("Loading data...\n")


    train_data = build_dataset(config)

    print("Train_set length: {}".format(len(train_data)))
    with open(config.print_path, 'a') as f:
        f.write("train_set: {}\n".format(len(train_data)))
    return train_data


def get_model(config):
    return PEAN_model_copy.PEAN(config).to(config.device)

UNK, PAD, CLS, SEP = '[UNK]', '[PAD]', '[CLS]','[SEP]'

def readMainPcapTLS(pcap_path, output_txt, labels):
    '''
    Assumes that all packets contain TLS, doesn't miss fragmented packets.
    '''
    from scapy.utils import PcapReader
    
    print(f"Converting {pcap_path} via Scapy")
    label = labels.get(os.path.basename(os.path.dirname(pcap_path)), "unknown") # defaults to unknown if label not found
    if label == 'unknown':
        raise ValueError(f'Couldn\'t find label in dict, looked for {os.path.basename(os.path.dirname(pcap_path))} and labels is {labels}')
    with open(output_txt, "a", encoding="utf-8") as f:
        len_seq = []
        pkt_bytes = []
        
        count = 0

        for pkt in PcapReader(pcap_path):
            count += 1
            # This takes both TCP layer and its payload 
            payload = bytes(pkt[TCP])
            len_seq.append(len(payload))
            pkt_bytes.append(payload[:400].hex(" "))

            if len(len_seq) == 15:
                f.write('\t'.join(pkt_bytes) + '\t')
                f.write(' '.join(map(str, len_seq)) + f'\t{label}\n')
                pkt_bytes = []
                len_seq = []

    print(f"Found {count} packets")
    print('Parsing finished!')

def readMainPcapScapy(pcap_path, output_txt, labels):
    from scapy.utils import PcapReader
    
    print(f"Converting {pcap_path} via Scapy")
    label = labels.get(os.path.basename(os.path.dirname(pcap_path)), "unknown") # defaults to unknown if label not found
    
    with open(output_txt, "a", encoding="utf-8") as f:
        len_seq = []
        pkt_bytes = []
        
        count = 0
        count_tls = 0

        for pkt in PcapReader(pcap_path):
            if not pkt.haslayer(TCP) or not pkt[TCP].payload:
                continue
            count += 1    
            payload = bytes(pkt[TCP].payload)
            
            # Simple TLS Check: TLS records start with 0x14, 0x15, 0x16, or 0x17
            # and the next two bytes are version (0x0301, 0x0303, etc.)
            if len(payload) >= 5 and payload[0] in [20, 21, 22, 23] and payload[1] == 3:
                try:
                    count_tls += 1
                    # 1. Calculate TLS record lengths (Equivalent to pkt.tls.record_length)
                    # TLS Header: [Type(1), Version(2), Length(2)]
                    tls_data_len = 0
                    pointer = 0
                    while pointer + 5 <= len(payload):
                        rec_len = int.from_bytes(payload[pointer+3:pointer+5], byteorder='big')
                        tls_data_len += rec_len
                        pointer += 5 + rec_len # Move to next record in same packet
                    
                    # 2. Total Length (TLS records + TCP len)
                    # pkt[TCP].len in pyshark is roughly the payload size
                    pkt_len = len(payload) 
                    
                    # 3. Format bytes (hex string, space separated, max 400 chars)
                    tls_hex = payload[:400].hex(" ") 
                    
                    len_seq.append(pkt_len)
                    pkt_bytes.append(tls_hex)
                    
                except Exception:
                    continue

            # Every 15 qualifying packets, write a row
            if len(len_seq) == 15:
                f.write('\t'.join(pkt_bytes) + '\t')
                f.write(' '.join(map(str, len_seq)) + f'\t{label}\n')
                pkt_bytes = []
                len_seq = []

    print(f"Found {count} packets")
    print(f"Out of them {count_tls} contained tls related bytes")
    print('Parsing finished!')



def readPcap_folderMain(folder_path, output_txt):
    # clean previous datasets
    if os.path.exists('TrafficData/class.txt'):
        os.remove('TrafficData/class.txt')
    if os.path.exists('TrafficData/sni_whs_train.txt'):
        os.remove('TrafficData/sni_whs_train.txt')
    if os.path.exists('DataCache/sni_whs_10_400_10.txt'):
        os.remove('DataCache/sni_whs_10_400_10.txt')

    labels = assign_labels(folder_path)
    for folder in os.listdir(folder_path):
        path = os.path.join(folder_path, folder)
        if not(os.path.isfile(path)):
            for file in os.listdir(path):
                #readPcapMain(os.path.join(path, file), output_txt, labels)
                #readMainPcapScapy(os.path.join(path, file), output_txt, labels)
                readMainPcapTLS(os.path.join(path, file), output_txt, labels)
    subprocess.run(['shuf', output_txt, '-o', output_txt])

def assign_labels(path):
    labels = {}
    for i, directory in enumerate(os.listdir(path)):
        labels[directory] = str(i)
    if not os.path.exists('TrafficData/class.txt'):
        print('Creating class.txt file')
        with open('TrafficData/class.txt', 'w') as f:
            for directory in labels:
                f.write(directory + ' ' + labels[directory] + '\n')
    else:
        print('Found already existing class.txt file, check if it comes from another dataset, then delete it and rerun program with --read_pcap flag')
    return labels

def build_dataset(config):
    def load_dataset(path, pad_num = 10, pad_length = 400, pad_len_seq = 10):
        cache_dir = './DataCache/'
        cached_dataset_file = cache_dir + '{}_{}_{}_{}.txt'.format(config.dataset, pad_num, pad_length, pad_len_seq)
        if os.path.exists(cached_dataset_file):
            print("Loading features from cached file {}".format(cached_dataset_file))
            with open(cached_dataset_file, "rb") as handle:
                contents = pickle.load(handle)
                return contents
        else:
            print("Creating training dataset....")
            contents = []
            with open(path, 'r') as f:
                for line in tqdm(f):
                    if not line:
                        continue
                    item = line.split('\t')
                    flow = item[0:-2]  # packets
                    if len(flow) < 2:
                        continue
                    if len(flow) > pad_num:
                        flow = flow[0 : pad_num]
                    length_seq = item[-2].strip().split(' ')
                    length_seq = list(map(int, length_seq))
                    label = item[-1]
                    masks = []
                    seq_lens = []
                    traffic_bytes_idss = []
                    for packet in flow:
                        traffic_bytes = config.tokenizer.tokenize(packet)
                        if len(traffic_bytes) <= pad_length - 2:
                            traffic_bytes = [CLS] + traffic_bytes + [SEP]
                        else:
                            traffic_bytes = [CLS] + traffic_bytes
                            traffic_bytes[pad_length - 1] = SEP


                        seq_len = len(traffic_bytes)
                        mask = []
                        traffic_bytes_ids = config.tokenizer.convert_tokens_to_ids(traffic_bytes)

                        if pad_length:
                            if len(traffic_bytes) < pad_length:
                                mask = [1] * len(traffic_bytes_ids) + [0] * (pad_length - len(traffic_bytes))  # [1,1,...,1,0,0]
                                traffic_bytes_ids += ([0] * (pad_length - len(traffic_bytes)))
                            else:
                                mask = [1] * pad_length
                                traffic_bytes_ids = traffic_bytes_ids[:pad_length]
                                seq_len = pad_length
                        traffic_bytes_idss.append(traffic_bytes_ids)
                        seq_lens.append(seq_len)
                        masks.append(mask)


                        if pad_len_seq:
                            if len(length_seq) < pad_len_seq:
                                length_seq += [0] * (pad_len_seq - len(length_seq))
                            else:
                                length_seq = length_seq[:pad_len_seq]


                    if pad_num: 
                        if len(traffic_bytes_idss) < pad_num:
                            len_tmp = len(traffic_bytes_idss)

                            mask = [0] * pad_length
                           
                            traffic_bytes_ids = [1] + [0] * (pad_length-2) + [2]
                            seq_len = 0
                            for i in range(pad_num - len_tmp):
                                masks.append(mask)
                                traffic_bytes_idss.append(traffic_bytes_ids)
                                seq_lens.append(seq_len)
                        else:
                            traffic_bytes_idss = traffic_bytes_idss[:pad_num]
                            masks = masks[:pad_num]
                            seq_lens = seq_lens[:pad_num]

                    contents.append((traffic_bytes_idss, seq_lens, masks, length_seq, int(label))) 

            print("Saving dataset cached file {}".format(cached_dataset_file))
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            with open(cached_dataset_file, "wb") as handle:
                pickle.dump(contents, handle, protocol=pickle.HIGHEST_PROTOCOL)
            return contents

    train = load_dataset(config.train_path, config.pad_num, config.pad_length, config.pad_len_seq)
    return train


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device, pad_len_seq):
        self.batch_size = batch_size 
        self.batches = batches
        print("len batches and batch size: ", len(batches), self.batch_size)
        self.n_batches = len(batches) // self.batch_size
        self.residue = False
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device
        self.pad_len_seq = pad_len_seq

    def _to_tensor(self, datas):
        # datas: batch_size * contents
        # contents: traffic_bytes_idss, seq_lens, masks, length_seq, int(label)
        traffic_bytes_idss = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        length_seq = torch.LongTensor([_[3] for _ in datas])
        length_seq = torch.reshape(length_seq, (-1,self.pad_len_seq,1)).to(self.device)
        seq_lens = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        masks = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        label = torch.LongTensor([_[4] for _ in datas]).to(self.device)

        return (traffic_bytes_idss, length_seq, seq_lens, masks), label

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device, config.pad_len_seq)
    return iter

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def check_data_variance(path):
    labels_dict = {}
    i = 0
    with open(path, 'r') as f:
        for line in f:
            i += 1
            if i == 1:
                l = line.split('\t')
                print(len(l[0:-2][1].split(' ')), len(l[-2].split(' ')), int(l[-1]))
            lab = line.strip().split('\t')[-1]
            if lab not in labels_dict:
                labels_dict[lab] = 0
            else:
                labels_dict[lab] += 1
    count = sum(labels_dict.values())
    return labels_dict, count

if __name__ == '__main__':
    print("Testing...")
    print(check_data_variance('TrafficData/sni_whs_train.txt'))
