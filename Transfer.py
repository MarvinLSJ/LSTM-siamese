#coding=utf-8
import os
import re
import sys
import yaml
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader

# utils
from utils import get_embedding, load_embed, save_embed, data_preprocessing, align_embeddings
# data
from data import myDS, mytestDS
# model
from model import Siamese_lstm

FLAGS = None


def main(_):
    # Load the configuration file.
    with open(FLAGS.config, 'r') as f:
        config = yaml.load(f)

    print('**********', config['experiment_name'],'**********')

    """ Cuda Check """
    if torch.cuda.is_available():
        print('Using GPU!')
    else: print('No GPU!')

    """
    Read Data
    """

    en = pd.read_csv("input/cleaned_en.csv")
    sp = pd.read_csv("input/cleaned_sp.csv")
    test_data = pd.read_csv("input/cleaned_test.csv")
    test_data.columns = ['s1', 's2']

    """ English """
    en.columns = ['s1', 's2', 'label']
    # split dataset
    msk = np.random.rand(len(en)) < 0.8
    en_train = en[msk]
    en_valid = en[~msk]
    en_all_sents = en['s1'].tolist() + en['s2'].tolist()

    # dataset
    en_trainDS = myDS(en_train, en_all_sents)
    en_validDS = myDS(en_valid, en_all_sents)

    """ Spanish """
    sp.columns = ['s1', 's2', 'label']
    # split dataset
    msk = np.random.rand(len(sp)) < 0.8
    sp_train = sp[msk]
    sp_valid = sp[~msk]
    sp_all_sents = sp['s1'].tolist() + sp['s2'].tolist()

    # dataset
    sp_trainDS = myDS(sp_train, sp_all_sents)
    sp_validDS = myDS(sp_valid, sp_all_sents)


    """
    Embedding
    """

    en_embed_dict = load_embed('input/en_embed.pkl')
    sp_embed_dict = load_embed('input/sp_embed.pkl')

    embed_size = 300
    en_embed_list = []
    for word in en_validDS.vocab._id2word:
        en_embed_list.append(en_embed_dict[word])
    en_vocab_size = len(en_embed_list)

    sp_embed_list = []
    for word in sp_trainDS.vocab._id2word:
        sp_embed_list.append(sp_embed_dict[word])
    sp_vocab_size = len(sp_embed_list)

    """ Align Embedding """
    aligned_size = max(en_vocab_size,sp_vocab_size)
    en_embedding = nn.Embedding(aligned_size, embed_size)
    sp_embedding = nn.Embedding(aligned_size, embed_size)

    en_embedding.weight, sp_embedding.weight = align_embeddings(en_embed_list, sp_embed_list, config['model']['embed_size'])


    """
    English Training
    """
    """ Model """
    config['embedding_matrix'] = en_embedding
    # model
    siamese_en = Siamese_lstm(config)
    print('English Model:',siamese_en)

    # loss func
    loss_weights = Variable(torch.FloatTensor([1, 3]))
    if torch.cuda.is_available():
        loss_weights = loss_weights.cuda()
    criterion = torch.nn.CrossEntropyLoss(loss_weights)

    # optimizer
    learning_rate = config['training']['en_learning_rate']
    if config['training']['en_optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, siamese_en.parameters()), lr=learning_rate)
    elif config['training']['en_optimizer'] == 'adam':
        optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, siamese_en.parameters()), lr=learning_rate)
    elif config['training']['en_optimizer'] == 'adadelta':
        optimizer = torch.optim.Adadelta(filter(lambda x: x.requires_grad, siamese_en.parameters()), lr=learning_rate)
    elif config['training']['en_optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(filter(lambda x: x.requires_grad, siamese_en.parameters()), lr=learning_rate)
    print('Optimizer:', config['training']['en_optimizer'])
    print('Learning rate:', config['training']['en_learning_rate'])

    # log info
    train_log_string = '%s :: Epoch %i :: Iter %i / %i :: train loss: %0.4f'
    valid_log_string = '%s :: Epoch %i :: valid loss: %0.4f\n'

    # Restore saved model (if one exists).
    ckpt_path = os.path.join(config['ckpt_dir'], config['experiment_name']+'-en.pt')
    if os.path.exists(ckpt_path):
        print('Loading checkpoint: %s' % ckpt_path)
        ckpt = torch.load(ckpt_path)
        epoch = ckpt['epoch']
        siamese_en.load_state_dict(ckpt['siamese'])
        optimizer.load_state_dict(ckpt['optimizer'])
    else:
        epoch = 1
        print('Fresh start!\n')


    if torch.cuda.is_available():
        criterion = criterion.cuda()
        siamese_en = siamese_en.cuda()

    """ English Model Train """

    if config['task'] == 'train':

        # save every epoch for visualization
        train_loss_record = []
        valid_loss_record = []
        best_record = 10.0

        # training
        print('Experiment: {}-English\n'.format(config['experiment_name']))

        while epoch < config['training']['num_epochs']:

            print('Start Epoch {} Training...'.format(epoch))

            # loss
            train_loss = []
            train_loss_sum = []
            # dataloader
            train_dataloader = DataLoader(dataset=en_trainDS, shuffle=True, num_workers=2, batch_size=1)

            for idx, data in enumerate(train_dataloader, 0):

                # get data
                s1, s2, label = data

                # clear gradients
                optimizer.zero_grad()

                # input
                output = siamese_en(s1, s2)
                output = output.squeeze(0)

                # label cuda
                label = Variable(label)
                if torch.cuda.is_available():
                    label = label.cuda()

                # loss backward
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.data.cpu())
                train_loss_sum.append(loss.data.cpu())

                # Every once and a while check on the loss
                if ((idx + 1) % 5000) == 0:
                    print(train_log_string % (datetime.now(), epoch, idx + 1, len(en_train), np.mean(train_loss)))
                    train_loss = []

            # Record at every epoch
            print('Train Loss at epoch {}: {}\n'.format(epoch, np.mean(train_loss_sum)))
            train_loss_record.append(np.mean(train_loss_sum))

            # Valid
            print('Epoch {} Validating...'.format(epoch))

            # loss
            valid_loss = []
            # dataloader
            valid_dataloader = DataLoader(dataset=en_validDS, shuffle=True, num_workers=2, batch_size=1)

            for idx, data in enumerate(valid_dataloader, 0):
                # get data
                s1, s2, label = data

                # input
                output = siamese_en(s1, s2)
                output = output.squeeze(0)

                # label cuda
                label = Variable(label)
                if torch.cuda.is_available():
                    label = label.cuda()

                # loss
                loss = criterion(output, label)
                valid_loss.append(loss.data.cpu())

            print(valid_log_string % (datetime.now(), epoch, np.mean(valid_loss)))
            # Record
            valid_loss_record.append(np.mean(valid_loss))
            epoch += 1

            if np.mean(valid_loss)-np.mean(train_loss_sum) > config['training']['earlystop']:
                 print("Early Stopping!")
                 break

            # Keep track of best record
            if np.mean(valid_loss) < best_record:
                best_record = np.mean(valid_loss)
                # save the best model
                state_dict = {
                    'epoch': epoch,
                    'siamese': siamese_en.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state_dict, ckpt_path)
                print('Model saved!\n')



    """
    Transfer to Spanish Model Training
    """

    """ Model """
    config['embedding_matrix'] = sp_embedding
    siamese_sp = Siamese_lstm(config)

    print('Spanish Model:', siamese_sp)

    # loss func
    loss_weights = Variable(torch.FloatTensor([1, 3]))
    if torch.cuda.is_available():
        loss_weights = loss_weights.cuda()
    criterion = torch.nn.CrossEntropyLoss(loss_weights)

    # optimizer
    learning_rate = config['training']['sp_learning_rate']
    if config['training']['sp_optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, siamese_sp.parameters()), lr=learning_rate)
    elif config['training']['sp_optimizer'] == 'adam':
        optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, siamese_sp.parameters()), lr=learning_rate)
    elif config['training']['sp_optimizer'] == 'adadelta':
        optimizer = torch.optim.Adadelta(filter(lambda x: x.requires_grad, siamese_sp.parameters()), lr=learning_rate)
    elif config['training']['sp_optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(filter(lambda x: x.requires_grad, siamese_sp.parameters()), lr=learning_rate)
    print('Optimizer:', config['training']['sp_optimizer'])
    print('Learning rate:', config['training']['sp_learning_rate'])

    ckpt_path = os.path.join(config['ckpt_dir'], config['experiment_name']+'-en.pt')
    print('Transfering English Model from: %s' % ckpt_path)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        siamese_sp.load_state_dict(ckpt['siamese'])
        epoch = 1
    else:
        print('No Transfer!')
        siamese_sp = Siamese_lstm(config)

    if torch.cuda.is_available():
        criterion = criterion.cuda()
        siamese_sp = siamese_sp.cuda()

    ckpt_path = os.path.join(config['ckpt_dir'], config['experiment_name'] + '-sp.pt')

    """ Spanish Model Train """

    if config['task'] == 'train':

        # save every epoch for visualization
        train_loss_record = []
        valid_loss_record = []
    #     best_record = 10.0

        # training
        print('Experiment: {}-Spanish\n'.format(config['experiment_name']))

        while epoch < config['training']['num_epochs']:

            print('Start Epoch {} Training...'.format(epoch))

            # loss
            train_loss = []
            train_loss_sum = []
            # dataloader
            train_dataloader = DataLoader(dataset=sp_trainDS, shuffle=True, num_workers=2, batch_size=1)

            for idx, data in enumerate(train_dataloader, 0):

                # get data
                s1, s2, label = data

                # clear gradients
                optimizer.zero_grad()

                # input
                output = siamese_sp(s1, s2)
                output = output.squeeze(0)

                # label cuda
                label = Variable(label)
                if torch.cuda.is_available():
                    label = label.cuda()

                # loss backward
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.data.cpu())
                train_loss_sum.append(loss.data.cpu())

                # Every once and a while check on the loss
                if ((idx + 1) % 5000) == 0:
                    print(train_log_string % (datetime.now(), epoch, idx + 1, len(sp_train), np.mean(train_loss)))
                    train_loss = []

            # Record at every epoch
            print('Train Loss at epoch {}: {}\n'.format(epoch, np.mean(train_loss_sum)))
            train_loss_record.append(np.mean(train_loss_sum))

            # Valid
            print('Epoch {} Validating...'.format(epoch))

            # loss
            valid_loss = []
            # dataloader
            valid_dataloader = DataLoader(dataset=sp_validDS, shuffle=True, num_workers=2, batch_size=1)

            for idx, data in enumerate(valid_dataloader, 0):
                # get data
                s1, s2, label = data

                # input
                output = siamese_sp(s1, s2)
                output = output.squeeze(0)

                # label cuda
                label = Variable(label)
                if torch.cuda.is_available():
                    label = label.cuda()

                # loss
                loss = criterion(output, label)
                valid_loss.append(loss.data.cpu())

            print(valid_log_string % (datetime.now(), epoch, np.mean(valid_loss)))
            # Record
            valid_loss_record.append(np.mean(valid_loss))
            epoch += 1

            if np.mean(valid_loss)-np.mean(train_loss_sum) > 0.02:
                 print("Early Stopping!")
                 break

            # Keep track of best record
            if np.mean(valid_loss) < best_record:
                best_record = np.mean(valid_loss)
                # save the best model
                state_dict = {
                    'epoch': epoch,
                    'siamese': siamese_sp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state_dict, ckpt_path)
                print('Model saved!\n')



    """ Inference """

    if config['task'] == 'inference':

        """ Load Spanish Model """

        ckpt_path = os.path.join(config['ckpt_dir'], config['experiment_name'] + '-sp.pt')
        if os.path.exists(ckpt_path):
            print('Loading checkpoint: %s' % ckpt_path)
            ckpt = torch.load(ckpt_path)
            siamese_sp.load_state_dict(ckpt['siamese'])
        else:
            print('Error Experiment Name!')

        testDS = mytestDS(test_data, sp_all_sents)
        # Do not shuffle here
        test_dataloader = DataLoader(dataset=testDS, num_workers=2, batch_size=1)

        result = []
        for idx, data in enumerate(test_dataloader, 0):

            # get data
            s1, s2 = data

            # input
            output = siamese_sp(s1,s2)
            output = output.squeeze(0)

            # feed output into softmax to get prob prediction
            sm = nn.Softmax(dim=1)
            res = sm(output.data)[:,1]
            result += res.data.tolist()

        result = pd.DataFrame(result)
        print(result.shape)
        print('Inference Done.')
        res_path = os.path.join(config['result']['filepath'], config['result']['filename'])
        result.to_csv(res_path, header=False, index=False)
        print('Result has writtn to', res_path, ', Good Luck!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Configuration file.')
    FLAGS, _ = parser.parse_known_args()
    main(_)






