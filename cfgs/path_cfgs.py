import os

class PATH:
    def __init__(self):

        # vqav2 dataset root path
        self.DATASET_PATH = {
            'vqa': './datasets/vqa/',
            'okvqa': './datasets/okvqa/',
            'aokvqa': './datasets/aokvqa/'
        }

        # bottom up features root path
        self.FEATURE_PATH = {
            'vqa': '/root/workspace/24s-VQA-MLLM/features',
            'okvqa': '/root/workspace/24s-VQA-MLLM/features',
            'aokvqa': '/root/workspace/24s-VQA-MLLM/features'
        }

        # Answer dictionary path
        self.ANSWER_DICT_PATH = {
            'vqa': '/root/workspace/24s-VQA-MLLM/BEiT3/assets/answer_dict_vqav2.json',
            'okvqa': '/root/workspace/24s-VQA-MLLM/BEiT3/assets/answer_dict_okvqa.json',
            'aokvqa': '/root/workspace/24s-VQA-MLLM/BEiT3/assets/answer_dict_aokvqa.json'
        }

        self.init_path()

    def init_path(self):
        self.IMG_FEAT_PATH = {
            'vqa': {
                'train': self.FEATURE_PATH['vqa'] + '/train2014/',
                'val': self.FEATURE_PATH['vqa'] + '/val2014/',
                'test': self.FEATURE_PATH['vqa'] + '/test2015/',
            },
            'okvqa': {
                'train': self.FEATURE_PATH['okvqa'] + '/train2014/',
                'test': self.FEATURE_PATH['okvqa'] + '/val2014/',
            },
            'aokvqa': {
                'train': self.FEATURE_PATH['aokvqa'] + '/train2017/',
                'val': self.FEATURE_PATH['aokvqa'] + '/val2017/',
                'test': self.FEATURE_PATH['aokvqa'] + '/test2017/',
            }
        }

        self.QUESTION_PATH = {
            'vqa': {
                'train': self.DATASET_PATH['vqa'] + 'v2_OpenEnded_mscoco_train2014_questions.json',
                'val': self.DATASET_PATH['vqa'] + 'v2_OpenEnded_mscoco_val2014_questions.json',
                'test': self.DATASET_PATH['vqa'] + 'v2_OpenEnded_mscoco_test2015_questions.json',
            },
            'okvqa': {
                'train': self.DATASET_PATH['okvqa'] + 'OpenEnded_mscoco_train2014_questions.json',
                'val': self.DATASET_PATH['okvqa'] + 'OpenEnded_mscoco_val2014_questions.json',
            },
            'aokvqa': {
                'train': self.DATASET_PATH['aokvqa'] + 'aokvqa_v1p0_train.json',
                'val': self.DATASET_PATH['aokvqa'] + 'aokvqa_v1p0_val.json',
                'test': self.DATASET_PATH['aokvqa'] + 'aokvqa_v1p0_test.json',
            }
        }

        self.ANSWER_PATH = {
            'vqa': {
                'train': self.DATASET_PATH['vqa'] + 'v2_mscoco_train2014_annotations.json',
                'val': self.DATASET_PATH['vqa'] + 'v2_mscoco_val2014_annotations.json',
            },
            'okvqa': {
                'train': self.DATASET_PATH['okvqa'] + 'mscoco_train2014_annotations.json',
                'val': self.DATASET_PATH['okvqa'] + 'mscoco_val2014_annotations.json',
            },
            'aokvqa': {
                'train': self.DATASET_PATH['aokvqa'] + 'aokvqa_v1p0_train_encoded.json',
                'val': self.DATASET_PATH['aokvqa'] + 'aokvqa_v1p0_val_encoded.json',
            }
        }

        self.RESULT_PATH = './results/result_test/'
        self.PRED_PATH = './results/pred/'
        self.CACHE_PATH = './results/cache/'
        self.LOG_PATH = './results/log/'
        self.CKPTS_PATH = './ckpts/'

        if 'result_test' not in os.listdir('./results'):
            os.mkdir('./results/result_test')

        if 'pred' not in os.listdir('./results'):
            os.mkdir('./results/pred')

        if 'cache' not in os.listdir('./results'):
            os.mkdir('./results/cache')

        if 'log' not in os.listdir('./results'):
            os.mkdir('./results/log')

        if 'ckpts' not in os.listdir('./'):
            os.mkdir('./ckpts')


    def check_path(self):
        print('Checking dataset ...')

        for dataset in self.IMG_FEAT_PATH:
            for mode in self.IMG_FEAT_PATH[dataset]:
                if not os.path.exists(self.IMG_FEAT_PATH[dataset][mode]):
                    print(f"{self.IMG_FEAT_PATH[dataset][mode]} NOT EXIST")
                    exit(-1)

            for mode in self.QUESTION_PATH[dataset]:
                if not os.path.exists(self.QUESTION_PATH[dataset][mode]):
                    print(f"{self.QUESTION_PATH[dataset][mode]} NOT EXIST")
                    exit(-1)

            for mode in self.ANSWER_PATH[dataset]:
                if not os.path.exists(self.ANSWER_PATH[dataset][mode]):
                    print(f"{self.ANSWER_PATH[dataset][mode]} NOT EXIST")
                    exit(-1)

        print('Finished')
        print('')

