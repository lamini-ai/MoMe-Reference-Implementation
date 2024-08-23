from transformers import LlamaConfig

class RaftConfig(LlamaConfig):
    model_type = "raft"

    def __init__(
        self,
        raft_embedding_batch_size=1,
        raft_embedding_seq_length=512,
        index_path="/Users/powerml/Desktop/MoME-Reference-Implementation/models/index",
        max_new_tokens=32,
        eval_results_path="/app/lamini-raft/data/eval_results.jsonl",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.raft_embedding_batch_size = raft_embedding_batch_size
        self.raft_embedding_seq_length = raft_embedding_seq_length
        self.index_path = index_path
        self.max_new_tokens = max_new_tokens
        self.eval_results_path = eval_results_path