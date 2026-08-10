import re


def _preprocess(text):
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def process_hellaswag_docs(dataset):
    def process_doc(doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        return {
            "query": _preprocess(doc["activity_label"] + ": " + ctx),
            "choices": [_preprocess(ending) for ending in doc["endings"]],
            "gold": int(doc["label"]),
        }

    return dataset.map(process_doc)


def hellaswag_doc_to_text(doc):
    return doc["query"]


def hellaswag_doc_to_choice(doc):
    return [" " + choice for choice in doc["choices"]]


def hellaswag_doc_to_target(doc):
    return doc["gold"]


def arc_doc_to_text(doc):
    return "Question: " + doc["question"] + "\nAnswer:"


def arc_doc_to_choice(doc):
    return [" " + choice for choice in doc["choices"]["text"]]


def arc_doc_to_target(doc):
    labels = ["A", "B", "C", "D", "E"]
    answer = str(doc["answerKey"])
    numeric_labels = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
    return labels.index(numeric_labels.get(answer, answer))


def piqa_doc_to_text(doc):
    return "Question: " + doc["goal"] + "\nAnswer:"


def piqa_doc_to_choice(doc):
    return [" " + doc["sol1"], " " + doc["sol2"]]


def piqa_doc_to_target(doc):
    return int(doc["label"])


def _gsm8k_solution(answer):
    solution, marker, final_answer = answer.partition("####")
    solution = re.sub(r"<<.*?>>", "", solution)
    solution = " ".join(solution.split())
    if marker:
        final_answer = final_answer.strip().rstrip(".")
        solution = f"{solution} So the answer is {final_answer}."
    return solution.strip()


def process_gsm8k_docs(dataset):
    def process_doc(doc):
        return {"_gold_solution": _gsm8k_solution(doc["answer"])}

    return dataset.map(process_doc)


def gsm8k_doc_to_text(doc):
    return "Question: " + doc["question"] + "\nAnswer:"


def gsm8k_doc_to_target(doc):
    return " " + doc["_gold_solution"]
