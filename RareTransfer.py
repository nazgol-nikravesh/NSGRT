import numpy as np
# H 测试样本分类结果  Test sample classification results
# TrainS 原训练样本 np数组  original training sample np array
# TrainA 辅助训练样本 Auxiliary training samples (the labeled diff-distribution data are treated as the auxiliary data)
# LabelS 原训练样本标签 Original training sample labels
# LabelA 辅助训练样本标签 Auxiliary training sample labels
# Test  测试样本 test sample
# N 迭代次数 number of iterations


class rareTransfer(object):
    def __init__(self, train_WC,  train_CC, label_WC, label_CC, testX, testY, N, initWeight, clf):
        self.train_WC = train_WC #target training
        self.train_CC = train_CC #source
        self.label_WC = label_WC
        self.label_CC = label_CC
        self.N = N
        self.test = testX
        self.test_l = testY
        self.weight = initWeight
        self.m = clf
        self.error = 0

    def fit(self):

        train_data = np.concatenate((self.train_CC, self.train_WC), axis=0)
        train_label = np.concatenate((self.label_CC, self.label_WC), axis=0)

        row_CC = self.train_CC.shape[0]
        row_WC = self.train_WC.shape[0]
        row_Test = self.test.shape[0]
        N = self.N

        test_data = np.concatenate((train_data, self.test), axis=0)

        # 初始化权重 Initialize weights
        weights_CC = self.weight.reshape(-1, 1)
        weights_WC = np.ones([row_WC, 1])*self.train_WC.shape[1]

        weights = np.concatenate((weights_CC, weights_WC), axis=0)



        # 防止除数为零 prevent division by zero
        if N == 0 or (1 + np.sqrt(2 * np.log(row_CC / N))) == 0:
            self.error = 1
            return
        beta = 1 / (1 + np.sqrt(np.log(2 * row_CC)/ N))

        # 存储每次迭代的标签和beta值？ Storing labels and beta values for each iteration?
        beta_T = np.zeros([1, N])
        result_label = np.ones([row_CC + row_WC + row_Test, N])

        predict = np.zeros([row_Test])

        # print('params initial finished.')
        train_data = np.asarray(train_data, order='C')
        train_label = np.asarray(train_label, order='C')
        test_data = np.asarray(test_data, order='C')

        for i in range(N):

            weights_WC= weights[row_CC:row_CC + row_WC, :] / np.sum(weights)
            weights_CC = weights[0:row_CC, :] / np.sum(weights)

            weights = np.concatenate((weights_CC, weights_WC), axis=0)

            P = self.calculate_P(weights, train_label)

            result_label[:, i] = self.train_classify(train_data, train_label, test_data, P)

            error_rate_source = self.calculate_error_rate(self.label_CC, result_label[0:row_CC, i], weights[0:row_CC, :])

            error_rate_target = self.calculate_error_rate(self.label_WC, result_label[row_CC:row_CC + row_WC, i], weights[row_CC:row_CC + row_WC, :])



            Cl = 1 - error_rate_source #label-dependent cost factor

            if error_rate_target >= 0.5:
                error_rate_target = 0.5
            beta_T[0, i] =  error_rate_target/(1 - error_rate_target)

            # 调整源域样本权重 Adjust original sample weights (target)
            for j in range(row_WC):
                weights[row_CC + j] = weights[row_CC + j] * np.power(beta_T[0, i],-1*np.abs(result_label[row_CC + j, i] - self.label_WC[j]))


            # 调整辅域样本权重 Adjust the sample weight of the auxiliary domain
            for j in range(row_CC):
                weights[j] = Cl*weights[j] * np.power(beta, np.abs(result_label[j, i] - self.label_CC[j]))


        for i in range(row_Test):
            # 跳过训练数据的标签 skip labels for training data
            left = np.sum(
                result_label[row_CC + row_WC + i, int(np.ceil(N / 2)):N] * np.log(1 / beta_T[0, int(np.ceil(N / 2)):N]))
            right = 0.5 * np.sum(np.log(1 / beta_T[0, int(np.ceil(N / 2)):N]))


            if left > right or left == right:
                predict[i] = 1
            else:
                predict[i] = 0

        self.label_p = predict

    def predict(self):
        return self.label_p, self.test_l


    def calculate_P(self, weights, label):
        total = np.sum(weights)
        return np.asarray(weights / total, order='C')


    def train_classify(self, train_data, train_label, test_data, P):
        train_data[train_data!=train_data] = 0
        train_label[train_label!=train_label] = 0
        test_data[test_data!=test_data] = 0
        P[P!=P] = 0

        self.m.fit(train_data, train_label, sample_weight=P[:, 0])

        return self.m.predict(test_data)


    def calculate_error_rate(self, label_R, label_H, weight):
        total = np.sum(weight)

        return np.sum(weight[:, 0]*np.abs(label_R - label_H)/ total)
