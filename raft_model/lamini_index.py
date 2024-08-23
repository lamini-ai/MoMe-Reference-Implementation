from lamini.api.embedding import Embedding

from tqdm import tqdm

import faiss
import json
import os
import numpy as np

import logging

logger = logging.getLogger(__name__)


class LaminiIndex:
    def __init__(self, loader=None, config={}, embedding_dimension=768):
        self.loader = loader
        self.config = config
        self.embedding_dimension = embedding_dimension
        if loader is not None:
            self.build_index()

    @staticmethod
    def load_index(path):
        lamini_index = LaminiIndex()

        faiss_path = os.path.join(path, "index.faiss")
        splits_path = os.path.join(path, "splits.json")
        keys_path = os.path.join(path, "keys.json")
        values_path = os.path.join(path, "values.json")

        # Load the index from a file
        lamini_index.index = faiss.read_index(faiss_path)

        # Load the splits from a file
        with open(splits_path, "r") as f:
            lamini_index.splits = json.load(f)

        # Load the key embeddings from a file
        with open(keys_path, "r") as f:
            lamini_index.keys = json.load(f)

        # Load the value embeddings from a file
        with open(keys_path, "r") as f:
            lamini_index.values = json.load(f)

        return lamini_index

    def build_index(self):
        self.splits = []

        self.index = None
        self.keys = []
        self.values = []

        # load a batch of splits from a generator
        for split_batch in tqdm(self.loader):
            embeddings = self.get_embeddings(split_batch)

            if self.index is None:
                # initialize the index
                logger.info(f"Creating index with dimension {len(embeddings[0])}")
                self.index = faiss.IndexFlatL2(len(embeddings[0]))

            # add the embeddings to the index
            self.index.add(np.array(embeddings))

            # save the splits
            self.splits.extend(split_batch)

            # save the key embeddings
            self.keys.extend(embeddings)

            # save the value embeddings
            self.values.extend(embeddings)

    def get_embeddings(self, examples):
        ebd = Embedding(config=self.config)
        embeddings = ebd.generate(examples)
        embedding_list = [embedding[0].tolist() for embedding in embeddings]

        embedding_list = self.truncate_embeddings(embedding_list)

        return embedding_list

    def truncate_embeddings(self, embeddings):
        return [embedding[: self.embedding_dimension] for embedding in embeddings]

    def get_key_and_value(self, query_embeddings, k=5):
        # get the k nearest neighbors
        distances, indices = self.index.search(query_embeddings, k)

        #print("loaded indices", indices)

        for i in indices:
            assert i[0] < len(
                self.keys
            ), f"Index {i[0]} not in keys, indices: {indices}, query_embeddings: {query_embeddings}"
            assert i[0] < len(
                self.values
            ), f"Index {i[0]} not in values, indices: {indices}"

        return (
            [self.keys[i[0]] for i in indices],
            [self.values[i[0]] for i in indices],
            [i[0] for i in indices],
        )

    def query(self, query, k=5):
        embedding = self.get_embeddings([query])[0]

        embedding_array = np.array([embedding])

        # get the k nearest neighbors
        distances, indices = self.index.search(embedding_array, k)

        return [self.splits[i] for i in indices[0]]

    def mmr_query(self, query, k=20, n=5):
        embedding = self.get_embeddings([query])[0]

        embedding_array = np.array([embedding])

        # get the k nearest neighbors
        distances, indices = self.index.search(embedding_array, k)

        # get the n most diverse results
        most_diverse = self.most_diverse_results(embedding, indices[0], n)

        return most_diverse

    def most_diverse_results(self, query_embedding, indices, n):
        # get the embeddings for the indices
        split_batch = [self.splits[i] for i in indices]

        embeddings = self.get_embeddings(split_batch)

        # calculate the similarity between the query and the results
        similarities = [np.dot(query_embedding, embedding) for embedding in embeddings]

        # initialize the results
        results = [indices[0]]

        # iterate through the results
        for i in range(1, n):
            # initialize the best result
            best_result = None
            best_result_similarity = 0

            # iterate through the remaining results
            for j in range(len(indices)):
                # skip the result if it is already in the results
                if indices[j] in results:
                    continue

                # calculate the similarity between the result and the other results
                similarity = np.mean(
                    [np.dot(embeddings[j], embeddings[k]) for k in range(len(results))]
                )

                # update the best result
                if similarity > best_result_similarity:
                    best_result = indices[j]
                    best_result_similarity = similarity

            # add the best result to the results
            results.append(best_result)

        return [self.splits[i] for i in results]

    def update(self, index, value):
        self.values[index] = value.tolist()

    def save_index(self, path):
        faiss_path = os.path.join(path, "index.faiss")
        splits_path = os.path.join(path, "splits.json")
        keys_path = os.path.join(path, "keys.json")
        values_path = os.path.join(path, "values.json")

        logger.debug("Saving index to %s", faiss_path)
        logger.debug("Saving splits to %s", splits_path)
        logger.debug("Saving key embeddings to %s", keys_path)
        logger.debug("Saving value embeddings to %s", values_path)

        logger.debug("Index size: %d", self.index.ntotal)

        # Save the index to a file
        faiss.write_index(self.index, faiss_path)

        # Save the splits to a file
        with open(splits_path, "w") as f:
            json.dump(self.splits, f)

        # Save the keys to a file
        with open(keys_path, "w") as f:
            json.dump(self.keys, f)

        # Save the values to a file
        with open(values_path, "w") as f:
            json.dump(self.values, f)