from stage4.eval_tasks import metrics, utils


class TinyDataset:
    def __init__(self, docs):
        self.docs = docs

    def map(self, function):
        return [function(doc) for doc in self.docs]


def test_basic_v2_prompt_and_target_conventions():
    hellaswag_doc = {
        "activity_label": "Roof shingle removal",
        "ctx_a": "A man is sitting on a roof.",
        "ctx_b": "he",
        "endings": ["is ripping tiles off.", "is holding a cube."],
        "label": "0",
    }
    processed = utils.process_hellaswag_docs(TinyDataset([hellaswag_doc]))
    doc = processed[0]

    assert doc["query"] == "Roof shingle removal: A man is sitting on a roof. He"
    assert utils.hellaswag_doc_to_choice(doc) == [
        " is ripping tiles off.",
        " is holding a cube.",
    ]
    assert utils.hellaswag_doc_to_target(doc) == 0

    gsm_doc = {
        "question": "How many?",
        "answer": "We calculate <<1+1=2>>2.\n#### 2",
    }
    processed_gsm = utils.process_gsm8k_docs(TinyDataset([gsm_doc]))[0]
    assert utils.gsm8k_doc_to_target(processed_gsm) == " We calculate 2. So the answer is 2."


def test_basic_v2_metrics_use_olmo_v2_length_rules():
    doc = {"choices": ["short", "much longer"], "gold": 1}
    results = [(-1.0, False), (-1.5, False)]

    output = metrics.process_len_norm_v2(doc, results)

    assert output["len_norm_v2"] == 1.0
    expected_bpb = 1.5 / len(" much longer".encode("utf-8")) * metrics.LOG_2_OF_E
    assert output["bpb_v2"] == expected_bpb


def test_basic_v2_piqa_metric_uses_solution_fields():
    doc = {"goal": "Choose", "sol1": "short", "sol2": "much longer", "label": 1}
    results = [(-1.0, False), (-1.5, False)]

    output = metrics.process_len_norm_v2(doc, results)

    assert output["len_norm_v2"] == 1.0
