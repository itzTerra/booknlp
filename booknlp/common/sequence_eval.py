import sys
import numpy as np
from booknlp.common.logger import get_logger


def check_span_f1_two_dicts_subcat(gold, pred):
    golds = {}
    preds = {}

    for target_lab in ["PRON", "NOM", "PROP"]:
        golds[target_lab] = {}
        preds[target_lab] = {}

        for doc, lab, start, end in gold:
            # print(lab)
            subcat = lab.split("_")[0]
            if subcat == target_lab:
                golds[target_lab][(doc, lab, start, end)] = 1

        for doc, lab, start, end in pred:
            subcat = lab.split("_")[0]
            if subcat == target_lab:
                preds[target_lab][(doc, lab, start, end)] = 1

    cor = 0.0
    for g in gold:
        if g in pred:
            cor += 1

    precision = 0
    if len(pred) > 0:
        precision = cor / len(pred)
    recall = 0
    if len(gold) > 0:
        recall = cor / len(gold)
    mainF = 0
    if (precision + recall) > 0:
        mainF = (2 * precision * recall) / (precision + recall)

    get_logger(enabled=True).info("precision: %.3f %s/%s" % (precision, cor, len(pred)))
    get_logger(enabled=True).info("recall: %.3f %s/%s" % (recall, cor, len(gold)))
    get_logger(enabled=True).info("F: %.3f" % mainF)

    for target_lab in ["PRON", "NOM", "PROP"]:
        cor = 0.0
        for g in golds[target_lab]:
            if g in preds[target_lab]:
                cor += 1

        precision = 0
        if len(preds[target_lab]) > 0:
            precision = cor / len(preds[target_lab])
        recall = 0
        if len(golds[target_lab]) > 0:
            recall = cor / len(golds[target_lab])
        F = 0
        if (precision + recall) > 0:
            F = (2 * precision * recall) / (precision + recall)

        get_logger(enabled=True).info(
            "\n\t%s precision: %.3f %s/%s"
            % (target_lab, precision, cor, len(preds[target_lab]))
        )
        get_logger(enabled=True).info(
            "\t%s recall: %.3f %s/%s"
            % (target_lab, recall, cor, len(golds[target_lab]))
        )
        get_logger(enabled=True).info("\t%s F: %.3f" % (target_lab, F))

    return mainF
