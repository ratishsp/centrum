import argparse
import nltk
import json
import torch
from rouge_score import rouge_scorer

DOCUMENT_SEPARATOR = "<doc-sep>"


def compute_rouge_scores(scorer, predictions, references):
    if isinstance(predictions, str):
        predictions = [predictions]
    if isinstance(references, str):
        references = [references]

    assert len(predictions) == len(references)
    all_scores = []
    for pred, ref in zip(predictions, references):
        all_scores.append(scorer.score(target=ref, prediction=pred))
    final_scores = {}
    for score_type in all_scores[0].keys():
        final_scores[score_type] = [
            all_scores[i][score_type] for i in range(len(all_scores))
        ]
    return final_scores


def compute_document_scores(cluster, summary, scorer):
    rouge_scores = []
    high_overlap = False
    for i_doc, doc in enumerate(cluster):
        scores = compute_rouge_scores(scorer, doc, summary)
        if scores["rougeL"][0].fmeasure > 0.8:
            high_overlap = True
        rouge_score = (
                          scores["rouge1"][0].fmeasure
                          + scores["rouge2"][0].fmeasure
                          + scores["rougeL"][0].fmeasure
                      ) / 3
        rouge_scores.append(rouge_score)
    return rouge_scores, high_overlap


def read_file(newshead_input_file_name):
    data = torch.load(newshead_input_file_name)
    return data


def get_scorer():
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True
    )
    return scorer


def clean_document(idx, chosen):
    paras = chosen.split("\n")
    cleaned = []
    for para in paras:
        if "This content is imported from" in para or "Invalid Email Something went wrong" in para or \
                        "Invalid email Sign up" in para or "When you subscribe we will use the information" in para \
                or "Get push notifications with news" in para or "Also Read" in para or "Read more:" in para \
                or "Advertisement Story continues below This advertisement has not loaded yet" in para \
                or "Please enter a valid email address." in para \
                or "This content is created and maintained" in para \
                or "Sorry, this video isn't available any more." in para \
                or "The video will auto-play soon" in para \
                or "Got a showbiz story?" in para \
                or "You can unsubscribe at any time" in para \
                or "Got a story for " in para \
                or "Click here to see " in para \
                or "This video is unavailable" in para \
                or "If you are using ad-blocking software" in para \
                or "Watch the video below" in para \
                or "Also read" in para:
            print(f"noisy para- index: {idx}, length: {len(para.split())}")
            if len(para.split()) > 50:
                print(">50")
            continue
        elif para.strip() in ["Getty Images", "Advertisement", "Related Gallery"]:
            continue
        cleaned.append(para.strip())
    return " ".join(cleaned)


def process(root_folder, output_root_folder, mode, index, max_input_length, max_output_length):
    data = read_file(f"{root_folder}/{mode}/newshead.{mode}.{index}.pt")
    scorer = get_scorer()
    cluster_file_name = f"{output_root_folder}/part/{mode}.{index}.jsonl"
    print(f"file_name: {cluster_file_name}")
    cluster_file = open(cluster_file_name, mode='w', encoding='utf-8', buffering=1)
    for count in range(len(data)):
        if count % 100 == 0:
            print(count)
        entry = data[count]
        if len(entry['articles']) < 3:
            continue
        scores = {}
        documents = [x['text'].strip() for x in entry['articles']]
        cleaned_documents = [clean_document(count, document) for document in documents]
        high_overlap = False
        # print(f"count: {count}")
        for doc_idx, article in enumerate(cleaned_documents):
            if len(article.split()) < 250:  # too short
                continue
            cluster = [x for x in cleaned_documents if x != article]
            document_rouge_scores, high_overlap = compute_document_scores(cluster, article, scorer)
            scores[doc_idx] = scores.get(doc_idx, 0) + sum(document_rouge_scores)
            if high_overlap:
                break
        if high_overlap:
            print(f"high overlap: {count}")
            continue

        if len(scores) < 2:  # we need to sort between as many candidate summaries as possible; 1 is too few
            continue

        max_key = max(scores, key=scores.get)
        chosen = cleaned_documents[max_key]

        input_cluster = [documents[key] for key in range(len(documents)) if key != max_key]
        if len(" ".join(input_cluster).split()) < 2 * len(
                documents[max_key].split()):  # ensuring there is compression required
            print(f"short cluster: {count}")
            continue

        chosen = truncate_doc(max_output_length, chosen, count)[0]
        input_cluster = truncate_doc(max_input_length, input_cluster, count)
        src = f"{DOCUMENT_SEPARATOR}".join(input_cluster) + f"{DOCUMENT_SEPARATOR}"
        example = {"index": count, "text": src, "summary": chosen}
        json.dump(example, cluster_file)
        cluster_file.write("\n")
    cluster_file.close()


def truncate_doc(max_length, documents, idx):
    truncated_docs = []
    summary = False
    if isinstance(documents, str):
        summary = True
        documents = [documents]
    for document in documents:
        truncated_doc = []
        paras = document.split("\n")
        cur_len = 0
        break_outer = False
        for para in paras:
            sents = nltk.sent_tokenize(para.strip())
            for sent in sents:
                if cur_len + len(sent.split()) > max_length // len(documents):
                    break_outer = True
                    break
                truncated_doc.append(sent.strip())
                cur_len += len(sent.split())
            if break_outer:
                break
        truncated_doc = " ".join(truncated_doc)
        if truncated_doc:
            truncated_docs.append(truncated_doc)
    return truncated_docs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='constructing dataset for pretraining')
    parser.add_argument('-root_folder', type=str,
                        help='path of root folder', default=None)
    parser.add_argument('-output_root_folder', type=str,
                        help='path of output root folder', default=None)
    parser.add_argument('-mode', type=str,
                        help='train/valid/test', default=None)
    parser.add_argument('-index', type=int,
                        help='index of the file', default=None)
    parser.add_argument('-max_input_length', type=int,
                        help='maximum length of input')
    parser.add_argument('-max_output_length', type=int,
                        help='maximum length of output')
    args = parser.parse_args()
    process(args.root_folder, args.output_root_folder, args.mode, args.index, args.max_input_length,
            args.max_output_length)
