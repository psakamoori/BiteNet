import sys
sys.path.append("./")

from utils.configs import cfg
from utils.record_log import RecordLog
import numpy as np
from BiteNet.model_mh import BiteNet as Model
import os
from dataset.dataset_full import VisitDataset
import warnings
import heapq
import operator
import tensorflow as tf
from tensorflow import keras 

from utils.evaluation import ConceptEvaluation as CodeEval, \
    EvaluationTemplate as Evaluation
warnings.filterwarnings('ignore')
logging = RecordLog()


from sklearn.metrics import top_k_accuracy_score

def train():

    visit_threshold = cfg.visit_threshold
    epochs = cfg.max_epoch
    batch_size = cfg.train_batch_size

    data_set = VisitDataset()
    data_set.prepare_data(visit_threshold)
    data_set.build_dictionary()
    data_set.load_data()
    code_eval = CodeEval(data_set, logging)
    print(data_set.train_context_codes.shape)
    print(data_set.train_intervals.shape)
    print(data_set.train_labels_2.shape)

    es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    # es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1)
    model = Model(data_set)
    model.build_network()
    print("shape of train_labels_l", data_set.train_labels_1.shape)
    print("shape of dev_labels_l", data_set.dev_labels_1.shape)

    model.model.fit(x=[data_set.train_context_codes,data_set.train_intervals],
                    y=data_set.train_labels_1,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=([data_set.dev_context_codes,data_set.dev_intervals], data_set.dev_labels_1)
                    # , callbacks=[es]
                    )

    #tf.keras.models.save_model(model, "../outputs",  overwrite=True,
    #                          include_optimizer=True,  save_format=None,
    #                          signatures=None, options=None, save_traces=True)

    metrics = model.model.evaluate([data_set.test_context_codes, data_set.test_intervals], data_set.test_labels_1)
    print("model.metrics_names =", model.model.metrics_names)
    log_str = 'Single fold accuracy is {}'.format(metrics[1])
    logging.add(log_str)

    trues = data_set.test_labels_1
    predicts = model.model.predict([data_set.test_context_codes, data_set.test_intervals])
    np.save('dx_true_labels', data_set.test_labels_1)
    np.save('dx_predict_labels', predicts)

    preVecs = []
    trueVecs = []
    for i in range(predicts.shape[0]):
        preVec = [rk[0] for rk in heapq.nlargest(30, enumerate(predicts[i]), key=operator.itemgetter(1))]
        preVecs.append(preVec)
        trueVec = [rk[0] for rk in
                   heapq.nlargest(np.count_nonzero(trues[i]), enumerate(trues[i]), key=operator.itemgetter(1))]
        trueVecs.append(trueVec)
    recalls = Evaluation.recall_top(trueVecs, preVecs)
    print("Recalls = ")
    logging.add(recalls)


    #from sklearn.preprocessing import MultiLabelBinarizer

    #tv = MultiLabelBinarizer().fit_transform(trueVecs)
    #pv = MultiLabelBinarizer().fit_transform(preVecs)

    #from sklearn.metrics import top_k_accuracy_score
    #pr_at_k = []
    #pr_at_k.append(top_k_accuracy_score(tv, pv, k=5))
    #pr_at_k.append(top_k_accuracy_score(tv, pv, k=10))
    #pr_at_k.append(top_k_accuracy_score(tv, pv, k=15))
    #pr_at_k.append(top_k_accuracy_score(tv, pv, k=20))
    #pr_at_k.append(top_k_accuracy_score(tv, pv, k=30))

    #logging.add(pr_at_k)
    #logging.done()

    #pr_accu = Evaluation.metric_pred(trueVecs, 0.5, preVecs)
    #logging.add(pr_accu)
    logging.done()

    embedding_weights = model.embedding.shared_weights
    embedding_values = tf.keras.backend.get_value(embedding_weights)

    icd__nmi = code_eval.get_clustering_nmi(embedding_values, 'ICD')
    logging.add('ICD, NMI Score: ' + str(icd__nmi))
    #ccs__nmi = code_eval.get_clustering_nmi(embedding_values, 'CCS')
    #logging.add('CCS, NMI Score: ' + str(ccs__nmi))

    for k in [1, 5, 10]:
        icd__nns = code_eval.get_nns_p_at_top_k(embedding_values, 'ICD', k)
        logging.add('ICD, nns Score: ' + str(icd__nns))
        #ccs__nns = code_eval.get_nns_p_at_top_k(embedding_values, 'CCS', k)
        #logging.add('CCS, nns Score: ' + str(ccs__nns))


def test():
    pass


def main():
    if cfg.train:
        train()
    else:
        test()


def output_model_params():
    logging.add()
    logging.add('==>model_title: ' + cfg.model_name[1:])
    logging.add()
    for key,value in cfg.args.__dict__.items():
        if key not in ['test','shuffle']:
            logging.add('%s: %s' % (key, value))


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)
    main()
