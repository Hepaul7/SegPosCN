import os
import re
from os import path
# TODO, Reference

OUTPUT_DIR = '../data'
if not path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)


class CTB7(object):
    """
    Data processor for Chinese Treebank CTB 7
    The files processed will be stored under ../data/CTB7
    Processing the POS-tagged (for labels)
    """

    def __init__(self):
        self.input_dir = './LDC10T07/data/utf-8/postagged'

        if not path.exists('./LDC10T07'):
            raise FileNotFoundError('CTB7 not found!')

        # 41-80,203-233,301-325,400-409,591,613-617,643-673,1022-1035,1120-1129,2110-2159,2270-
        # 2294,2510-2569,2760-2799,3040-3109,4040-4059,4084-4085,4090,4096,4106-4108,4113-
        # 4115,4121,4128,4132,4135,4158-4162,4169,4189,4196,4236-4261,4322,4335-4336,4407-4411
        dev_index = [i for i in range(41, 81)]
        dev_index.extend([i for i in range(203, 234)])
        dev_index.extend([i for i in range(301, 326)])
        dev_index.extend([i for i in range(400, 410)])
        dev_index.extend([591])
        dev_index.extend([i for i in range(613, 618)])
        dev_index.extend([i for i in range(643, 674)])
        dev_index.extend([i for i in range(1022, 1036)])
        dev_index.extend([i for i in range(1120, 1130)])
        dev_index.extend([i for i in range(2110, 2160)])
        dev_index.extend([i for i in range(2270, 2295)])
        dev_index.extend([i for i in range(2510, 2570)])
        dev_index.extend([i for i in range(2760, 2800)])
        dev_index.extend([i for i in range(3040, 3110)])
        dev_index.extend([i for i in range(4040, 4060)])
        dev_index.extend([i for i in range(4084, 4086)])
        dev_index.extend([4090, 4096])
        dev_index.extend([i for i in range(4106, 4109)])
        dev_index.extend([i for i in range(4113, 4116)])
        dev_index.extend([4121, 4128, 4132, 4135, 4169, 4189, 4196, 4322])
        dev_index.extend([i for i in range(4158, 4163)])
        dev_index.extend([i for i in range(4236, 4262)])
        dev_index.extend([i for i in range(4335, 4337)])
        dev_index.extend([i for i in range(4407, 4412)])

        self.dev_index = set(dev_index)

        # 1-40,144-174,271-300,410-428,592,900-931,1009-1020,1036,1044,1060-1061,1072,1118-1119,1132,1141-
        # 1142,1148,2000-2010,2160-2220,2295-2330,2570-2640,2800-2845,3110-3145,4030-4039,4060-
        # 4070,4086-4087,4091,4097,4109-4112,4118-4120,4127,4133-4134,4136-4139,4163-4168,4188,4197-
        # 4235,4321,4334,4337,4400-4406
        test_index = [i for i in range(1, 41)]
        test_index.extend([i for i in range(144, 175)])
        test_index.extend([i for i in range(271, 301)])
        test_index.extend([i for i in range(410, 429)])
        test_index.extend([i for i in range(900, 932)])
        test_index.extend([i for i in range(1009, 1021)])
        test_index.extend(
            [592, 1036, 1044, 1060, 1061, 1072, 1118, 1119, 1132, 1141, 1142, 1148, 4091, 4097, 4127, 4133,
             4134, 4188, 4321, 4334, 4337])
        test_index.extend([i for i in range(2000, 2011)])
        test_index.extend([i for i in range(2160, 2221)])
        test_index.extend([i for i in range(2295, 2331)])
        test_index.extend([i for i in range(2570, 2641)])
        test_index.extend([i for i in range(2800, 2846)])
        test_index.extend([i for i in range(3110, 3146)])
        test_index.extend([i for i in range(4030, 4040)])
        test_index.extend([i for i in range(4060, 4071)])
        test_index.extend([i for i in range(4086, 4088)])
        test_index.extend([i for i in range(4109, 4113)])
        test_index.extend([i for i in range(4118, 4121)])
        test_index.extend([i for i in range(4136, 4140)])
        test_index.extend([i for i in range(4163, 4169)])
        test_index.extend([i for i in range(4197, 4236)])
        test_index.extend([i for i in range(4400, 4407)])

        for i in test_index:
            if i in dev_index:
                raise ValueError()

        self.test_index = set(test_index)

    def process(self):
        print('processing %s' % str(self.input_dir))

        input_dir = self.input_dir
        output_dir = path.join(OUTPUT_DIR, 'CTB7')
        if not path.exists(output_dir):
            os.mkdir(output_dir)
        train = []
        test = []
        dev = []
        input_file_list = os.listdir(input_dir)
        input_file_list.sort()
        label2id = {}
        index = 1
        for file_name in input_file_list:
            if not file_name.endswith('.pos'):
                continue
            file_path = path.join(input_dir, file_name)

            file_index = int(file_name[file_name.find('_') + 1: file_name.rfind('.') - 3])

            with open(file_path, 'r', encoding='utf8') as f:
                data = []
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if line == '' or line.startswith('<'):
                        continue
                    word_list = re.split('\\s+', line)
                    for word in word_list:
                        item = word.split('_')
                        if item[0] == '':
                            continue
                        word = item[0]
                        pos = item[1]
                        if len(word) == 1:
                            prefix = ['S']
                        elif len(word) == 2:
                            prefix = ['B', 'E']
                        else:
                            prefix = ['B']
                            for i in range(len(word) - 2):
                                prefix.append('M')
                            prefix.append('E')
                        assert len(prefix) == len(word)
                        for char, pre in zip(word, prefix):
                            data.append('%s\t%s' % (char, pre + '-' + pos))
                            if pre + '-' + pos not in label2id:
                                label2id[pre + '-' + pos] = index
                                index += 1
                    data.append('\n')
                if file_index in self.test_index:
                    test.extend(data)
                elif file_index in self.dev_index:
                    dev.extend(data)
                else:
                    train.extend(data)

        output_train_path = path.join(output_dir, 'train.tsv')
        output_test_path = path.join(output_dir, 'test.tsv')
        output_dev_path = path.join(output_dir, 'dev.tsv')
        output_label2id_path = path.join(output_dir, 'label2id')

        with open(output_train_path, 'w', encoding='utf8') as f:
            for line in train:
                f.write(line)
                f.write('\n')
        with open(output_dev_path, 'w', encoding='utf8') as f:
            for line in dev:
                f.write(line)
                f.write('\n')
        with open(output_test_path, 'w', encoding='utf8') as f:
            for line in test:
                f.write(line)
                f.write('\n')
        with open(output_label2id_path, 'w', encoding='utf8') as f:
            for lable, id in label2id.items():
                f.write('%s\t%d' % (lable, id))
                f.write('\n')


class CTB9(object):
    # Data split follows https://www.aclweb.org/anthology/I17-1018/ (Table 10)
    def __init__(self):
        self.input_dir = './LDC2016T13/data/postagged'

        if not path.exists('./LDC2016T13'):
            raise FileNotFoundError('CTB9 not found!')

        # 0044 - 0143, 0170 - 0270, 0400 - 0899,
        # 1001 - 1017, 1019, 1021 - 1035, 1037 - 1043,
        # 1045 - 1059, 1062 - 1071, 1073 - 1117,
        # 1120 - 1131, 1133 - 1140, 1143 - 1147,
        # 1149 - 1151, 2000 - 2915, 4051 - 4099,
        # 4112 - 4180, 4198 - 4368, 5000 - 5446,
        # 6000 - 6560, 7000 - 7013
        train_index = [i for i in range(44, 144)]
        train_index.extend([i for i in range(170, 271)])
        train_index.extend([i for i in range(400, 900)])
        train_index.extend([i for i in range(1001, 1018)])
        train_index.extend([1019])
        train_index.extend([i for i in range(1021, 1036)])
        train_index.extend([i for i in range(1037, 1044)])
        train_index.extend([i for i in range(1045, 1060)])
        train_index.extend([i for i in range(1062, 1072)])
        train_index.extend([i for i in range(1073, 1118)])
        train_index.extend([i for i in range(1120, 1132)])
        train_index.extend([i for i in range(1133, 1141)])
        train_index.extend([i for i in range(1143, 1148)])
        train_index.extend([i for i in range(1149, 1152)])
        train_index.extend([i for i in range(2000, 2916)])
        train_index.extend([i for i in range(4051, 4100)])
        train_index.extend([i for i in range(4112, 4181)])
        train_index.extend([i for i in range(4198, 4369)])
        train_index.extend([i for i in range(5000, 5447)])
        train_index.extend([i for i in range(6000, 6561)])
        train_index.extend([i for i in range(7000, 7014)])

        self.train_index = set(train_index)

        # 0301 - 0326, 2916 - 3030, 4100 - 4106,
        # 4181 - 4189, 4369 - 4390, 5447 - 5492,
        # 6561 - 6630, 7013 - 7014

        # 41-80,203-233,301-325,400-409,591,613-617,643-673,1022-1035,1120-1129,2110-2159,2270-
        # 2294,2510-2569,2760-2799,3040-3109,4040-4059,4084-4085,4090,4096,4106-4108,4113-
        # 4115,4121,4128,4132,4135,4158-4162,4169,4189,4196,4236-4261,4322,4335-4336,4407-4411
        dev_index = [i for i in range(301, 327)]
        dev_index.extend([i for i in range(2916, 3031)])
        dev_index.extend([i for i in range(4100, 4107)])
        dev_index.extend([i for i in range(4181, 4190)])
        dev_index.extend([i for i in range(4369, 4391)])
        dev_index.extend([i for i in range(5447, 5493)])
        dev_index.extend([i for i in range(6561, 6631)])
        dev_index.extend([i for i in range(7014, 7015)])

        self.dev_index = set(dev_index)

        # 0001 - 0043, 0144 - 0169, 0271 - 0301,
        # 0900 - 0931, 1018, 1020, 1036, 1044,
        # 1060, 1061, 1072, 1118, 1119, 1132,
        # 1141, 1142, 1148, 3031 - 3145,
        # 4107 - 4111, 4190 - 4197, 4391 - 4411,
        # 5493 - 5558, 6631 - 6700, 7015 - 7017

        # 1-40,144-174,271-300,410-428,592,900-931,1009-1020,1036,1044,1060-1061,1072,1118-1119,1132,1141-
        # 1142,1148,2000-2010,2160-2220,2295-2330,2570-2640,2800-2845,3110-3145,4030-4039,4060-
        # 4070,4086-4087,4091,4097,4109-4112,4118-4120,4127,4133-4134,4136-4139,4163-4168,4188,4197-
        # 4235,4321,4334,4337,4400-4406
        test_index = [i for i in range(1, 44)]
        test_index.extend([i for i in range(144, 170)])
        test_index.extend([i for i in range(271, 301)])
        test_index.extend([i for i in range(900, 932)])
        test_index.extend([1018, 1020, 1036, 1044, 1060, 1061, 1072, 1118, 1119, 1132, 1141, 1142, 1148])
        test_index.extend([i for i in range(3031, 3146)])
        test_index.extend([i for i in range(4107, 4112)])
        test_index.extend([i for i in range(4190, 4198)])
        test_index.extend([i for i in range(4391, 4412)])
        test_index.extend([i for i in range(5493, 5559)])
        test_index.extend([i for i in range(6631, 6701)])
        test_index.extend([i for i in range(7015, 7018)])

        for i in test_index:
            if i in dev_index:
                print(i)
                raise ValueError()
            if i in train_index:
                print(i)
                raise ValueError()
        for i in dev_index:
            if i in train_index:
                print(i)
                raise ValueError()

        self.test_index = set(test_index)

    def process(self):

        print('processing %s' % str(self.input_dir))

        input_dir = self.input_dir
        output_dir = path.join(OUTPUT_DIR, 'CTB9')
        if not path.exists(output_dir):
            os.mkdir(output_dir)
        train = []
        test = []
        dev = []
        input_file_list = os.listdir(input_dir)
        input_file_list.sort()
        label2id = {}
        index = 1
        for file_name in input_file_list:
            if not file_name.endswith('.pos'):
                continue
            file_path = path.join(input_dir, file_name)

            file_index = int(file_name[file_name.find('_') + 1: file_name.rfind('.') - 3])

            with open(file_path, 'r', encoding='utf8') as f:
                data = []
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if line == '' or line.startswith('<'):
                        continue
                    word_list = re.split('\\s+', line)
                    for word in word_list:
                        item = word.split('_')
                        if item[0] == '':
                            continue
                        word = item[0]
                        pos = item[1]
                        if len(word) == 1:
                            prefix = ['S']
                        elif len(word) == 2:
                            prefix = ['B', 'E']
                        else:
                            prefix = ['B']
                            for i in range(len(word) - 2):
                                prefix.append('M')
                            prefix.append('E')
                        assert len(prefix) == len(word)
                        for char, pre in zip(word, prefix):
                            data.append('%s\t%s' % (char, pre + '-' + pos))
                            if pre + '-' + pos not in label2id:
                                label2id[pre + '-' + pos] = index
                                index += 1
                    data.append('\n')
                if file_index in self.test_index:
                    test.extend(data)
                elif file_index in self.dev_index:
                    dev.extend(data)
                elif file_index in self.train_index:
                    train.extend(data)

        output_train_path = path.join(output_dir, 'train.tsv')
        output_test_path = path.join(output_dir, 'test.tsv')
        output_dev_path = path.join(output_dir, 'dev.tsv')
        output_label2id_path = path.join(output_dir, 'label2id')

        with open(output_train_path, 'w', encoding='utf8') as f:
            for line in train:
                f.write(line)
                f.write('\n')
        with open(output_dev_path, 'w', encoding='utf8') as f:
            for line in dev:
                f.write(line)
                f.write('\n')
        with open(output_test_path, 'w', encoding='utf8') as f:
            for line in test:
                f.write(line)
                f.write('\n')
        with open(output_label2id_path, 'w', encoding='utf8') as f:
            for lable, id in label2id.items():
                f.write('%s\t%d' % (lable, id))
                f.write('\n')

        # print(label2id.keys())
