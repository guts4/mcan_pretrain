# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from core.data.data_utils import img_feat_path_load, img_feat_load, ques_load, tokenize, ans_stat
from core.data.data_utils import proc_img_feat, proc_ques, proc_ans

import numpy as np
import glob, json, torch, time
import torch.utils.data as Data
import os


class DataSet(Data.Dataset):
    def __init__(self, __C):
        self.__C = __C
        self.test_as_val = False

        # --------------------------
        # ---- Raw data loading ----
        # --------------------------

        # Loading all image paths
        self.img_feat_path_list = []
        self.ques_list = []
        self.ans_list = []
        split_list = __C.SPLIT[__C.RUN_MODE].split('+')
        split_list = [split for split in split_list if split in ['train', 'val', 'test']]

        for split in split_list:
            # okvqa의 경우 'val'을 'test'로 대체
            if split == 'val' and 'val' not in __C.IMG_FEAT_PATH:
                split = 'test'
                self.test_as_val = True  # test를 val로 사용 중임을 표시

            try:
                self.img_feat_path_list += glob.glob(__C.IMG_FEAT_PATH[split] + '*.npz')
                print(f"Success {__C.IMG_FEAT_PATH[split]}")
            except KeyError:
                print(f"Warning: No path found for split '{split}' in dataset '{__C.DATASET}'. Skipping.")
                continue

        # Loading question word list
        self.stat_ques_list = []
        for split in split_list:
            if split == 'val' and 'val' not in __C.QUESTION_PATH:
                split = 'test'

            try:
                # aokvqa의 경우 별도로 처리
                # if __C.DATASET == 'aokvqa':
                #     self.stat_ques_list += json.load(open(__C.QUESTION_PATH[split], 'r'))
                # else:
                #     self.stat_ques_list += json.load(open(__C.QUESTION_PATH[split], 'r'))['questions']
                self.stat_ques_list += json.load(open('/root/datasets/okvqa/data/vqav2/v2_OpenEnded_mscoco_train2014_questions.json', 'r'))['questions']
                self.stat_ques_list += json.load(open('/root/datasets/okvqa/data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json', 'r'))['questions']
                self.stat_ques_list += json.load(open('/root/workspace/24s-VQA-MLLM/dataset/vqav2/v2_OpenEnded_mscoco_test2015_questions.json', 'r'))['questions']
                self.stat_ques_list += json.load(open('/root/datasets/okvqa/data/okvqa/OpenEnded_mscoco_train2014_questions.json', 'r'))['questions']
                self.stat_ques_list += json.load(open('/root/datasets/okvqa/data/okvqa/OpenEnded_mscoco_val2014_questions.json', 'r'))['questions']
                self.stat_ques_list += json.load(open('/root/datasets/okvqa/data/aokvqa/aokvqa_v1p0_train.json', 'r'))
                self.stat_ques_list += json.load(open('/root/datasets/okvqa/data/aokvqa/aokvqa_v1p0_val.json', 'r'))
                self.stat_ques_list += json.load(open('/root/datasets/okvqa/data/aokvqa/aokvqa_v1p0_test.json', 'r'))
                
            except KeyError:
                print(f"Warning: No question path found for split '{split}' in dataset '{__C.DATASET}'. Skipping.")
                continue

        # Loading answer word list
        self.ques_list = []
        self.ans_list = []
        for split in split_list:
            if split == 'val' and 'val' not in __C.QUESTION_PATH:
                split = 'test'

            try:
                if __C.DATASET == 'aokvqa':
                    self.ques_list += json.load(open(__C.QUESTION_PATH[split], 'r'))
                    if __C.RUN_MODE in ['train']:
                        self.ans_list += json.load(open(__C.ANSWER_PATH[split], 'r'))
                else:
                    self.ques_list += json.load(open(__C.QUESTION_PATH[split], 'r'))['questions']
                    if __C.RUN_MODE in ['train']:
                        self.ans_list += json.load(open(__C.ANSWER_PATH[split], 'r'))['annotations']
            except KeyError:
                print(f"Warning: No answer path found for split '{split}' in dataset '{__C.DATASET}'. Skipping.")
                continue

        # Define run data size
        if __C.RUN_MODE in ['train']:
            self.data_size = len(self.ans_list)
        else:
            self.data_size = len(self.ques_list)

        print('== Dataset size:', self.data_size)


        # ------------------------
        # ---- Data statistic ----
        # ------------------------

        # {image id} -> {image feature absolutely path}
        if self.__C.PRELOAD:
            print('==== Pre-Loading features ...')
            time_start = time.time()
            self.iid_to_img_feat = img_feat_load(self.img_feat_path_list)
            time_end = time.time()
            print('==== Finished in {}s'.format(int(time_end-time_start)))
        else:
            self.iid_to_img_feat_path = img_feat_path_load(self.img_feat_path_list)

        # {question id} -> {question}
        self.qid_to_ques = ques_load(self.ques_list)

        # Tokenize
        self.token_to_ix, self.pretrained_emb = tokenize(self.stat_ques_list, __C.USE_GLOVE)
        self.token_size = self.token_to_ix.__len__()
        print('== Question token vocab size:', self.token_size)

        # Answers statistic
        # Make answer dict during training does not guarantee
        # the same order of {ans_to_ix}, so we published our
        # answer dict to ensure that our pre-trained model
        # can be adapted on each machine.

        # Thanks to Licheng Yu (https://github.com/lichengunc)
        # for finding this bug and providing the solutions.

        # self.ans_to_ix, self.ix_to_ans = ans_stat(self.stat_ans_list, __C.ANS_FREQ)
        self.ans_to_ix, self.ix_to_ans = self.load_answer_dict("/root/workspace/24s-VQA-MLLM/BEiT3/assets/answer_dict_aokvqa.json")
        self.ans_size = self.ans_to_ix.__len__()
        print('== Answer vocab size (occurr more than {} times):'.format(8), self.ans_size)
        print('Finished!')
        print('')

    def load_answer_dict(self, answer_dict_path):
        # Load the answer dictionary and convert to ans_to_ix and ix_to_ans
        with open(answer_dict_path, 'r') as f:
            answers = json.load(f)

        ans_to_ix = {ans: ix for ix, ans in enumerate(answers)}
        ix_to_ans = {str(ix): ans for ix, ans in enumerate(answers)}

        return ans_to_ix, ix_to_ans

    def __getitem__(self, idx):

        # For code safety
        img_feat_iter = np.zeros(1)
        ques_ix_iter = np.zeros(1)
        ans_iter = np.zeros(1)

        # Process ['train'] and ['val', 'test'] respectively
        if self.__C.RUN_MODE in ['train']:
            # Load the run data from list
            ans = self.ans_list[idx]
            ques = self.qid_to_ques[str(ans['question_id'])]

            # Process image feature from (.npz) file
            if self.__C.PRELOAD:
                img_feat_x = self.iid_to_img_feat[str(ans['image_id'])]
            else:
                img_feat = np.load(self.iid_to_img_feat_path[str(ans['image_id'])])
                #img_feat_x = img_feat['x'].transpose((1, 0))
                img_feat_x = img_feat['x']
            #img_feat_iter = proc_img_feat(img_feat_x, self.__C.IMG_FEAT_PAD_SIZE)
            img_feat_iter = img_feat['x']

            # Process question
            ques_ix_iter = proc_ques(ques, self.token_to_ix, self.__C.MAX_TOKEN)

            # Process answer
            ans_iter = proc_ans(ans, self.ans_to_ix)

        else:
            # Load the run data from list
            ques = self.ques_list[idx]

            # # Process image feature from (.npz) file
            # img_feat = np.load(self.iid_to_img_feat_path[str(ques['image_id'])])
            # img_feat_x = img_feat['x'].transpose((1, 0))
            # Process image feature from (.npz) file
            if self.__C.PRELOAD:
                img_feat_x = self.iid_to_img_feat[str(ques['image_id'])]
            else:
                img_feat = np.load(self.iid_to_img_feat_path[str(ques['image_id'])])
                # img_feat_x = img_feat['x'].transpose((1, 0))
                img_feat_x = img_feat['x']
            # img_feat_iter = proc_img_feat(img_feat_x, self.__C.IMG_FEAT_PAD_SIZE)
            img_feat_iter = img_feat['x']

            # Process question
            ques_ix_iter = proc_ques(ques, self.token_to_ix, self.__C.MAX_TOKEN)


        return torch.from_numpy(img_feat_iter), \
               torch.from_numpy(ques_ix_iter), \
               torch.from_numpy(ans_iter)


    def __len__(self):
        return self.data_size


