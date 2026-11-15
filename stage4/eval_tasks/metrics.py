import numpy as np


LOG_2_OF_E = 1.4426950408889634


def _choices(doc):
    if "choices" in doc:
        choices = doc["choices"]
        if isinstance(choices, dict):
            choices = choices["text"]
    else:
        choices = [doc["sol1"], doc["sol2"]]
    return [" " + choice for choice in choices]


def _gold(doc):
    if "gold" in doc:
        return doc["gold"]
    if "label" in doc and "goal" in doc:
        return int(doc["label"])
    labels = ["A", "B", "C", "D", "E"]
    answer = str(doc["answerKey"])
    numeric_labels = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
    return labels.index(numeric_labels.get(answer, answer))


def _scores(doc, results):
    choices = _choices(doc)
    loglikelihoods = np.asarray([float(result[0]) for result in results])
    byte_lengths = np.asarray([len(choice.encode("utf-8")) for choice in choices])
    gold = _gold(doc)
    bpb = -loglikelihoods[gold] / byte_lengths[gold] * LOG_2_OF_E
    return choices, loglikelihoods, gold, bpb


def process_acc_v2(doc, results):
    _, loglikelihoods, gold, bpb = _scores(doc, results)
    return {"acc_v2": float(np.argmax(loglikelihoods) == gold), "bpb_v2": float(bpb)}


def process_len_norm_v2(doc, results):
    choices, loglikelihoods, gold, bpb = _scores(doc, results)
    char_lengths = np.asarray([len(choice) for choice in choices])
    return {
        "len_norm_v2": float(np.argmax(loglikelihoods / char_lengths) == gold),
        "bpb_v2": float(bpb),
    }


def process_gsm8k_bpb_v2(doc, results):
    loglikelihood = float(results[0][0])
    target = " " + doc["_gold_solution"]
    bpb = -loglikelihood / len(target.encode("utf-8")) * LOG_2_OF_E
    return {"bpb_v2": float(bpb)}
