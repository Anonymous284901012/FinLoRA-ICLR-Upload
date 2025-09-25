import xml.etree.ElementTree as ET
import re
import json
from typing import List, Dict
from tqdm import tqdm
import random
import os.path
from huggingface_hub import HfApi
import csv

random.seed(42)
import subprocess

example_qa_dict = {
    "xbrl_tags": {
        "q": "What is the US GAAP XBRL tag for Cash and Cash Equivalents as reported by Example Company Inc for the Fiscal Year ending in FY 2022",
        "a": "us-gaap:AnExampleTagName"
    },
    "value": {
        "q": "What is the value of Exapmle company's income for the Fiscal year ending in FY 2020?",
        "a": "80000000"
    },
    "formula_calculation": {
        "q": "Can you provide the formula for Operating Profit Margin from Example Corp for the Fiscal Year ending in FY 2022?",
        "a": "(50000000 / 3590000000) * 100"
    },
    "formula_formatted_with_tags": {
        "q": "What is the formula for the Gross Profit Margin of Example Inc, formatted with the relevant US GAAP XBRL tags",
        "a": "us-gaap:ExampleTag / us-gaap:AnotherExampleTag) * 100"
    }
}


def get_xbrl_dataset(data: List[Dict], cat):
    """
    Saves entries with matching category1 or category2 in the format for fine-tuning.

    Args:
        data (List[Dict]): The input JSON data.
        category (str): The category name to match.
    """
    results = []
    for entry in data:
        question = entry["query"]
        question = re.sub(r"\(.*?\)", "", question)
        context_ids = entry["contextID"]

        # if not os.path.isfile('train/DowJones30/' + doc_path):
        #     print(f"missing file {doc_path}")
        #     continue

        example_qa = f"Example question: {example_qa_dict[cat]['q']}\nExample answer: {example_qa_dict[cat]['a']}"
        output = entry["raw_answer"]

        if cat == 'formula_calculation':
            question += " Answer with a formula substituted with values. "
            output = entry["value_formula_answer"]

        output = str(output)
        instruction = (f"You are a knowledgeable XBRL assistant. Your task is to analyze the XBRL context and provide an"
                       f" accurate and very concise answer to the question, The example question can help you to learn the "
                       f"answer format. DO NOT output xml, code, explanation or create new question. \n{example_qa}\n")

        input = f"Question: {question}\nAnswer:"
        year = int(entry["{fiscal year/quarter}"].replace("FY ", ""))

        # context_xml = add_xml(instruction + input, doc_path}, context_ids[0])
        # if len(context_xml) > 24000:
        #     continue

        # print(entry["answer"])
        # entry["doc_path"], entry["answer"], entry["contextID"][0]
        results.append({
            "instruction": instruction,
            "input": input,
            "output": output,
            "year": year,
            "company": entry["ticker"],
            "doc_path": entry['doc_path'],
            "context_id": context_ids[0],
        })

    print("final length", len(results))
    return results


def gen_xbrl(cat):
    with open("xbrl_bench_34020.json", "r") as f:
        data = json.load(f)
        filtered_data = [entry for entry in data if entry['category1'] == cat or entry['category2'] == cat]

        all_doc_path = list(set([entry['doc_path'] for entry in filtered_data]))
        print(f"Total data size for {cat}: {len(filtered_data)}, total number of filings {len(all_doc_path)}")
        random.shuffle(filtered_data)

        dataset = get_xbrl_dataset(filtered_data, cat)

        test_data = [x for x in dataset if x['year'] == 2023]
        train_data = [x for x in dataset if x['year'] != 2023]

        print(f"train size: {len(train_data)}, test size: {len(test_data)}\n")

        # with open(f"{cat}_test.csv", "w", newline="") as f:
        #     w = csv.DictWriter(f, test_data[0].keys(), quoting=csv.QUOTE_ALL)
        #     w.writeheader()
        #     w.writerows(test_data)
        # with open(f"{cat}_train.csv", "w", newline="") as f:
        #     w = csv.DictWriter(f, train_data[0].keys(), quoting=csv.QUOTE_ALL)
        #     w.writeheader()
        #     w.writerows(train_data)

        return train_data, test_data


if __name__ == '__main__':
    tags_train, tags_test = gen_xbrl("xbrl_tags")
    value_train, value_test = gen_xbrl("value")
    formula_train, formula_test = gen_xbrl("formula_formatted_with_tags")
    formula_calc_train, formula_calc_test = gen_xbrl("formula_calculation")
