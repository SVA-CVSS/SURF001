import os
import csv
import torch
from transformers import RobertaTokenizer, RobertaModel
import re


class feature_extraction:
    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained('microsoft/graphcodebert-base')
        self.model = RobertaModel.from_pretrained('microsoft/graphcodebert-base')
        self.input_folder = "../data/before1"
        self.output_file = "../data/graphcode_feature.csv"
        self.output_folder = "../data/text"

    def process_file(self, file_path):
        if not file_path.endswith(".dot"):
            return [], []
        print('Processing file:', file_path)

        with open(file_path, "r", encoding='utf-8') as f:
            content = f.read()
        node_pattern = re.compile(r'"(\d+)" \[label = <(.*?)> \]')
        edge_pattern = re.compile(r'"(\d+)"\s*->\s*"(\d+)"\s*\[\s*label\s*=\s*"([^"]*)"\s*\]')
        # edge_pattern = re.compile(r'"(\d+)"\s*->\s*"(\d+)"\s*')

        nodes = []
        edges = []

        for match in node_pattern.finditer(content):
            node_id = int(match.group(1))
            node_label = {"label": match.group(2)}
            nodes.append((node_id, node_label))

        for match in edge_pattern.finditer(content):
            source_id = int(match.group(1))
            target_id = int(match.group(2))
            edge_label = {"label": match.group(3)}
            edges.append((source_id, target_id, edge_label))

        return nodes, edges

    def process_folder(self, folder_path):
        print('Processing folder:', folder_path)
        all_nodes = []
        all_edges = []
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                nodes, edges = self.process_file(file_path)
                all_nodes.extend(nodes)
                all_edges.extend(edges)
        return all_nodes, all_edges

    def write_text_to_file(self, text_list, folder_path, file_name):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            for text in text_list:
                f.write(text + "\n")

    def process(self):
        tokenizer = self.tokenizer
        model = self.model
        input_folder = self.input_folder
        output_file = self.output_file
        output_folder = self.output_folder

        with open(output_file, "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["filename", "embedding"])

            for filename in os.listdir(input_folder):
                if filename.endswith(".c"):
                    with open(os.path.join(input_folder, filename), "r", encoding='utf-8') as f:
                        content = f.read()

                        nodes_list, edges_list = self.process_folder('../data/cpg/' + filename[:-2])

                        nodes_set = set()
                        edges_set = set()

                        nodes_text = []
                        for node_id, node_attr in nodes_list:
                            node_text = f"{node_id}-{node_attr['label']}"
                            if node_text not in nodes_set:
                                nodes_set.add(node_text)
                                nodes_text.append(node_text)

                        edges_text = []
                        for source_id, target_id, edge_attr in edges_list:
                            edge_text = f"{source_id}-{target_id}-{edge_attr['label']}"
                            # edge_text = f"{source_id}-{target_id}"
                            if edge_text not in edges_set:
                                edges_set.add(edge_text)
                                edges_text.append(edge_text)

                        nodes_output_folder = os.path.join(output_folder, "nodes")
                        edges_output_folder = os.path.join(output_folder, "edges")

                        os.makedirs(nodes_output_folder, exist_ok=True)
                        os.makedirs(edges_output_folder, exist_ok=True)

                        self.write_text_to_file(nodes_text, nodes_output_folder, f"{filename[:-2]}_nodes.txt")
                        self.write_text_to_file(edges_text, edges_output_folder, f"{filename[:-2]}_edges.txt")

                        full_input_text = f"<code> {content}, <graph> {' '.join(nodes_text)} {' '.join(edges_text)} "
                        # full_input_text = f"<graph> {' '.join(nodes_text)} {' '.join(edges_text)}"

                    encoded_input = tokenizer.encode_plus(full_input_text, return_tensors='pt', max_length=512)

                    with torch.no_grad():
                        outputs = model(**encoded_input)

                    last_hidden_state = outputs[0]
                    embedding = torch.mean(last_hidden_state, dim=1).squeeze()

                    writer.writerow([filename[:-2], embedding.tolist()])
