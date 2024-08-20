# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from core.data.load_data import DataSet
from core.model.net import Net
from core.model.optim import get_optim, adjust_lr
from core.data.data_utils import shuffle_list
from utils.vqa import VQA
from utils.vqaEval import VQAEval

import os, json, torch, datetime, pickle, copy, shutil, time
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import ConcatDataset

class Execution:
    def __init__(self, __C):
        self.__C = __C
        self.datasets = []

        """print('Loading training set ........')
        for dataset_name in __C.DATASET_LIST:
            __C_specific = copy.deepcopy(__C)
            setattr(__C_specific, 'DATASET', dataset_name)
            setattr(__C_specific, 'RUN_MODE', 'train')

            # 올바른 경로 설정
            __C_specific.IMG_FEAT_PATH = __C.IMG_FEAT_PATH[dataset_name]
            __C_specific.QUESTION_PATH = __C.QUESTION_PATH[dataset_name]
            __C_specific.ANSWER_PATH = __C.ANSWER_PATH.get(dataset_name, None)

            print(f'Loading {dataset_name} dataset...')
            dataset = DataSet(__C_specific)
            
            # 추가 사항
            dataset.name = dataset_name
            self.datasets.append(dataset)

        # Concatenate datasets into one
        self.dataset = torch.utils.data.ConcatDataset(self.datasets)"""

        self.dataset_eval = None
        __C.EVAL_EVERY_EPOCH = True
        if __C.EVAL_EVERY_EPOCH:
            self.datasets_eval = []
            for dataset_name in __C.DATASET_LIST:
                __C_eval = copy.deepcopy(__C)
                setattr(__C_eval, 'DATASET', dataset_name)
                setattr(__C_eval, 'RUN_MODE', 'val')

                # 올바른 경로 설정
                __C_eval.IMG_FEAT_PATH = __C.IMG_FEAT_PATH[dataset_name]
                __C_eval.QUESTION_PATH = __C.QUESTION_PATH[dataset_name]
                __C_eval.ANSWER_PATH = __C.ANSWER_PATH.get(dataset_name, None)

                print(f'Loading {dataset_name} validation set for per-epoch evaluation...')
                dataset_eval = DataSet(__C_eval)

                # 추가 사항
                dataset_eval.name = dataset_name
                self.datasets_eval.append(dataset_eval)

            # Concatenate evaluation datasets into one
            self.dataset_eval = torch.utils.data.ConcatDataset(self.datasets_eval)


    def train(self, dataset, dataset_eval=None):
        # 기존의 필요한 정보 획득 부분은 그대로 유지합니다.
        data_size = sum([len(ds) for ds in dataset.datasets])
        token_size = max([ds.token_size for ds in dataset.datasets])
        ans_size = max([ds.ans_size for ds in dataset.datasets])
        pretrained_emb = dataset.datasets[0].pretrained_emb
        # print('data_size:', data_size)
        # print('token_size:', token_size)
        # print('ans_size:', ans_size)
        # print('pretrained_emb:', pretrained_emb)
        # exit()

        # MCAN 모델 정의 부분도 그대로 유지합니다.
        net = Net(
            self.__C,
            pretrained_emb,
            token_size,
            ans_size
        )
        net.cuda()
        net.train()

        # 다중 GPU 학습을 위한 설정 부분도 유지합니다.
        if self.__C.N_GPU > 1:
            net = nn.DataParallel(net, device_ids=self.__C.DEVICES)

        # BCE 로스 정의 부분도 유지합니다.
        loss_fn = torch.nn.BCELoss(reduction='sum').cuda()

        # Checkpoint 로드 부분도 유지합니다.
        if self.__C.RESUME:
            print(' ========== Resume training')
            if self.__C.CKPT_PATH is not None:
                path = self.__C.CKPT_PATH
            else:
                path = self.__C.CKPTS_PATH + \
                    'ckpt_' + self.__C.CKPT_VERSION + \
                    '/epoch' + str(self.__C.CKPT_EPOCH) + '.pkl'

            print('Loading ckpt {}'.format(path))
            ckpt = torch.load(path)
            print('Finish!')
            net.load_state_dict(ckpt['state_dict'])
            optim = get_optim(self.__C, net, data_size, ckpt['lr_base'])
            optim._step = int(data_size / self.__C.BATCH_SIZE * self.__C.CKPT_EPOCH)
            optim.optimizer.load_state_dict(ckpt['optimizer'])
            start_epoch = self.__C.CKPT_EPOCH
        else:
            if ('ckpt_' + self.__C.VERSION) in os.listdir(self.__C.CKPTS_PATH):
                shutil.rmtree(self.__C.CKPTS_PATH + 'ckpt_' + self.__C.VERSION)
            os.mkdir(self.__C.CKPTS_PATH + 'ckpt_' + self.__C.VERSION)
            optim = get_optim(self.__C, net, data_size)
            start_epoch = 0

        loss_sum = 0
        named_params = list(net.named_parameters())
        grad_norm = np.zeros(len(named_params))

        # Shuffle the ans_list in each dataset if external shuffle is set
        if self.__C.SHUFFLE_MODE in ['external']:
            dataloader = Data.DataLoader(
                dataset,
                batch_size=self.__C.BATCH_SIZE,
                shuffle=False,
                num_workers=self.__C.NUM_WORKERS,
                pin_memory=self.__C.PIN_MEM,
                drop_last=True
            )
        else:
            dataloader = Data.DataLoader(
                dataset,
                batch_size=self.__C.BATCH_SIZE,
                shuffle=True,
                num_workers=self.__C.NUM_WORKERS,
                pin_memory=self.__C.PIN_MEM,
                drop_last=True
            )

        # Training script
        for epoch in range(start_epoch, self.__C.MAX_EPOCH):

            # Save log information
            logfile = open(
                self.__C.LOG_PATH +
                'log_run_' + self.__C.VERSION + '.txt',
                'a+'
            )
            logfile.write(
                'nowTime: ' +
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
                '\n'
            )
            logfile.close()

            # Learning Rate Decay
            if epoch in self.__C.LR_DECAY_LIST:
                adjust_lr(optim, self.__C.LR_DECAY_R)

            # Externally shuffle
            if self.__C.SHUFFLE_MODE == 'external':
                for ds in dataset.datasets:
                    shuffle_list(ds.ans_list)

            time_start = time.time()
            # Iteration
            for step, (
                    img_feat_iter,
                    ques_ix_iter,
                    ans_iter
            ) in enumerate(dataloader):

                optim.zero_grad()

                img_feat_iter = img_feat_iter.cuda()
                ques_ix_iter = ques_ix_iter.cuda()
                ans_iter = ans_iter.cuda()

                for accu_step in range(self.__C.GRAD_ACCU_STEPS):

                    sub_img_feat_iter = \
                        img_feat_iter[accu_step * self.__C.SUB_BATCH_SIZE:
                                      (accu_step + 1) * self.__C.SUB_BATCH_SIZE]
                    sub_ques_ix_iter = \
                        ques_ix_iter[accu_step * self.__C.SUB_BATCH_SIZE:
                                     (accu_step + 1) * self.__C.SUB_BATCH_SIZE]
                    sub_ans_iter = \
                        ans_iter[accu_step * self.__C.SUB_BATCH_SIZE:
                                 (accu_step + 1) * self.__C.SUB_BATCH_SIZE]
                    
                    print('sub_img_feat_iter:', img_feat_iter.shape)
                    print('sub_ques_ix_iter:', ques_ix_iter.shape)
                    exit()


                    pred = net(
                        sub_img_feat_iter,
                        sub_ques_ix_iter
                    )

                    loss = loss_fn(pred, sub_ans_iter)
                    # only mean-reduction needs be divided by grad_accu_steps
                    # removing this line wouldn't change our results because the speciality of Adam optimizer,
                    # but would be necessary if you use SGD optimizer.
                    # loss /= self.__C.GRAD_ACCU_STEPS
                    loss.backward()
                    loss_sum += loss.cpu().data.numpy() * self.__C.GRAD_ACCU_STEPS

                    if self.__C.VERBOSE:
                        if dataset_eval is not None:
                            mode_str = self.__C.SPLIT['train'] + '->' + self.__C.SPLIT['val']
                        else:
                            mode_str = self.__C.SPLIT['train'] + '->' + self.__C.SPLIT['test']

                        print("\r[version %s][epoch %2d][step %4d/%4d][%s] loss: %.4f, lr: %.2e" % (
                            self.__C.VERSION,
                            epoch + 1,
                            step,
                            int(data_size / self.__C.BATCH_SIZE),
                            mode_str,
                            loss.cpu().data.numpy() / self.__C.SUB_BATCH_SIZE,
                            optim._rate
                        ), end='          ')

                # Gradient norm clipping
                if self.__C.GRAD_NORM_CLIP > 0:
                    nn.utils.clip_grad_norm_(
                        net.parameters(),
                        self.__C.GRAD_NORM_CLIP
                    )

                # Save the gradient information
                for name in range(len(named_params)):
                    norm_v = torch.norm(named_params[name][1].grad).cpu().data.numpy() \
                        if named_params[name][1].grad is not None else 0
                    grad_norm[name] += norm_v * self.__C.GRAD_ACCU_STEPS
                    # print('Param %-3s Name %-80s Grad_Norm %-20s'%
                    #       (str(grad_wt),
                    #        params[grad_wt][0],
                    #        str(norm_v)))

                optim.step()

            time_end = time.time()
            print('Finished in {}s'.format(int(time_end-time_start)))

            # print('')
            epoch_finish = epoch + 1

            # Save checkpoint
            state = {
                'state_dict': net.state_dict(),
                'optimizer': optim.optimizer.state_dict(),
                'lr_base': optim.lr_base
            }
            torch.save(
                state,
                self.__C.CKPTS_PATH +
                'ckpt_' + self.__C.VERSION +
                '/epoch' + str(epoch_finish) +
                '.pkl'
            )

            # Logging
            logfile = open(
                self.__C.LOG_PATH +
                'log_run_' + self.__C.VERSION + '.txt',
                'a+'
            )
            logfile.write(
                'epoch = ' + str(epoch_finish) +
                '  loss = ' + str(loss_sum / data_size) +
                '\n' +
                'lr = ' + str(optim._rate) +
                '\n\n'
            )
            logfile.close()

            # Eval after every epoch
            if dataset_eval is not None:
                self.eval(
                    dataset_eval,
                    state_dict=net.state_dict(),
                    valid=True
                )

            # if self.__C.VERBOSE:
            #     logfile = open(
            #         self.__C.LOG_PATH +
            #         'log_run_' + self.__C.VERSION + '.txt',
            #         'a+'
            #     )
            #     for name in range(len(named_params)):
            #         logfile.write(
            #             'Param %-3s Name %-80s Grad_Norm %-25s\n' % (
            #                 str(name),
            #                 named_params[name][0],
            #                 str(grad_norm[name] / data_size * self.__C.BATCH_SIZE)
            #             )
            #         )
            #     logfile.write('\n')
            #     logfile.close()

            loss_sum = 0
            grad_norm = np.zeros(len(named_params))


    def eval(self, dataset, state_dict=None, valid=False):
        # Load parameters
        """if self.__C.CKPT_PATH is not None:
            print('Warning: you are now using CKPT_PATH args, '
                'CKPT_VERSION and CKPT_EPOCH will not work')

            path = self.__C.CKPT_PATH
        else:
            path = self.__C.CKPTS_PATH + \
                'ckpt_' + self.__C.CKPT_VERSION + \
                '/epoch' + str(self.__C.CKPT_EPOCH) + '.pkl'

        val_ckpt_flag = False
        if state_dict is None:
            val_ckpt_flag = True
            print('Loading ckpt {}'.format(path))
            state_dict = torch.load(path)['state_dict']
            print('Finish!')

        qid_list = []
        for sub_dataset in dataset.datasets:
            qid_list.extend([ques['question_id'] for ques in sub_dataset.ques_list])
        ans_ix_list = []
        pred_list = []

        data_size = sum([sub_dataset.data_size for sub_dataset in dataset.datasets])
        token_size = dataset.datasets[0].token_size  # Assuming all datasets have the same token size
        ans_size = dataset.datasets[0].ans_size  # Assuming all datasets have the same ans_size
        pretrained_emb = dataset.datasets[0].pretrained_emb

        net = Net(
            self.__C,
            pretrained_emb,
            token_size,
            ans_size
        )
        net.cuda()
        net.eval()

        if self.__C.N_GPU > 1:
            net = nn.DataParallel(net, device_ids=self.__C.DEVICES)

        net.load_state_dict(state_dict)

        dataloader = Data.DataLoader(
            dataset,
            batch_size=self.__C.EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=self.__C.NUM_WORKERS,
            pin_memory=True
        )

        # Separate results for each dataset
        vqa_results = []
        okvqa_results = []
        aokvqa_results = []

        # Modified loop to go through the entire dataset
        for step, (img_feat_iter, ques_ix_iter, ans_iter) in enumerate(dataloader):
            print("\rEvaluation: [step %4d/%4d]" % (
                step,
                int(data_size / self.__C.EVAL_BATCH_SIZE),
            ), end='          ')

            img_feat_iter = img_feat_iter.cuda()
            ques_ix_iter = ques_ix_iter.cuda()

            pred = net(img_feat_iter, ques_ix_iter)
            pred_np = pred.cpu().data.numpy()
            pred_argmax = np.argmax(pred_np, axis=1)

            # Save the answer index
            if pred_argmax.shape[0] != self.__C.EVAL_BATCH_SIZE:
                pred_argmax = np.pad(
                    pred_argmax,
                    (0, self.__C.EVAL_BATCH_SIZE - pred_argmax.shape[0]),
                    mode='constant',
                    constant_values=-1
                )

            ans_ix_list.append(pred_argmax)  # Use extend to accumulate results for all steps

            # Save the whole prediction vector
            if self.__C.TEST_SAVE_PRED:
                if pred_np.shape[0] != self.__C.EVAL_BATCH_SIZE:
                    pred_np = np.pad(
                        pred_np,
                        ((0, self.__C.EVAL_BATCH_SIZE - pred_np.shape[0]), (0, 0)),
                        mode='constant',
                        constant_values=-1
                    )

                pred_list.append(pred_np)

        print('')
        ans_ix_list = np.array(ans_ix_list).reshape(-1)

        print(f"qid_list size: {len(qid_list)}")
        start_idx = 0
        for sub_dataset in dataset.datasets:
            sub_data_size = len(sub_dataset.ques_list)
            print(f"{sub_dataset.name} ques_list size: {len(sub_dataset.ques_list)}")
            for qix in range(start_idx, start_idx + sub_data_size):
                qid = qid_list[qix]
                try:
                    qid = int(qid)  # 문자열을 정수로 변환 시도
                except ValueError:
                    pass  # 변환이 실패한 경우 그대로 사용

                answer_dict = {
                    'answer': sub_dataset.ix_to_ans[str(ans_ix_list[qix])],
                    'question_id': qid
                }

                if sub_dataset.name == 'vqa':
                    vqa_results.append(answer_dict)
                elif sub_dataset.name == 'okvqa':
                    okvqa_results.append(answer_dict)
                elif sub_dataset.name == 'aokvqa':
                    aokvqa_results.append(answer_dict)

            start_idx += sub_data_size

        # Save each result separately
        with open(self.__C.RESULT_PATH + 'vqa_results.json', 'w') as f:
            json.dump(vqa_results, f)
        with open(self.__C.RESULT_PATH + 'okvqa_results.json', 'w') as f:
            json.dump(okvqa_results, f)
        with open(self.__C.RESULT_PATH + 'aokvqa_results.json', 'w') as f:
            json.dump(aokvqa_results, f)"""

        # Run validation script for each dataset if valid is True
        if valid:
            # Validate VQA
            ques_file_path = self.__C.QUESTION_PATH['vqa']['val']
            ans_file_path = self.__C.ANSWER_PATH['vqa']['val']

            vqa = VQA(ans_file_path, ques_file_path)
            vqaRes = vqa.loadRes(self.__C.RESULT_PATH + 'vqa_results.json', ques_file_path)
            vqaEval = VQAEval(vqa, vqaRes, n=2)
            vqaEval.evaluate()

            print("\nOverall Accuracy for VQA is: %.02f\n" % (vqaEval.accuracy['overall']))
            print("Per Answer Type Accuracy for VQA is the following:")
            for ansType in vqaEval.accuracy['perAnswerType']:
                print("%s : %.02f" % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
            print("\n")

            # Validate OKVQA
            ques_file_path = self.__C.QUESTION_PATH['okvqa']['val']
            ans_file_path = self.__C.ANSWER_PATH['okvqa']['val']
            vqa = VQA(ans_file_path, ques_file_path)
            vqaRes = vqa.loadRes(self.__C.RESULT_PATH + 'okvqa_results.json', ques_file_path)
            vqaEval = VQAEval(vqa, vqaRes, n=2)
            vqaEval.evaluate()

            print("\nOverall Accuracy for OKVQA is: %.02f\n" % (vqaEval.accuracy['overall']))
            print("Per Answer Type Accuracy for OKVQA is the following:")
            for ansType in vqaEval.accuracy['perAnswerType']:
                print("%s : %.02f" % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
            print("\n")

            # Validate AOKVQA
            ques_file_path = self.__C.QUESTION_PATH['aokvqa']['val']
            ans_file_path = self.__C.ANSWER_PATH['aokvqa']['val']
            vqa = VQA(ans_file_path, ques_file_path)
            vqaRes = vqa.loadRes(self.__C.RESULT_PATH + 'aokvqa_results.json', ques_file_path)
            vqaEval = VQAEval(vqa, vqaRes, n=2)
            vqaEval.evaluate()

            print("\nOverall Accuracy for AOKVQA is: %.02f\n" % (vqaEval.accuracy['overall']))
            print("Per Answer Type Accuracy for AOKVQA is the following:")
            for ansType in vqaEval.accuracy['perAnswerType']:
                print("%s : %.02f" % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
            print("\n")




    def run(self, run_mode):
        if run_mode == 'train':
            self.empty_log(self.__C.VERSION)
            self.train(self.dataset, self.dataset_eval)

        elif run_mode == 'val':
            self.eval(self.dataset_eval, valid=True)

        elif run_mode == 'test':
            self.eval(self.dataset)

        else:
            exit(-1)


    def empty_log(self, version):
        print('Initializing log file ........')
        if (os.path.exists(self.__C.LOG_PATH + 'log_run_' + version + '.txt')):
            os.remove(self.__C.LOG_PATH + 'log_run_' + version + '.txt')
        print('Finished!')
        print('')




