from unstructured.partition.pdf import partition_pdf
import os
import pickle

file_path = "./Attention_is_all_you_need.pdf"
cache_file = "./chunks_cache.pkl"

def get_chunks():
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    chunks = partition_pdf(
        filename=file_path,
        strategy="hi_res",
        hi_res_model_name="yolox",
        extract_image_block_types=["Image"],

        extract_image_block_to_payload=True,

        chunking_strategy="by_title",
        max_characters=6000,
        combine_text_under_n_chars=1200,
        new_after_n_chars=3600,
    )

    with open(cache_file, "wb") as f:
        pickle.dump(chunks, f)

    return chunks

chunks = get_chunks()