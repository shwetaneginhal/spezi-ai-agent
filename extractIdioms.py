import os
import json



def save_chunks_to_jsonl(generator, output_file="semantic_chunks.jsonl"):
    print(f"Writing chunks to {output_file}...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk, meta in generator:
            # Combine the chunk text and metadata into a single dictionary
            record = {
                "semantic_chunk": chunk,
                "metadata": meta
            }
            # Write as a single line of valid JSON
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    print("Successfully saved all chunks!")


def generate_semantic_chunks(repo_dir, split="idiom_trainplus"):
    """
    Simultaneously reads parallel German text, English text, and info files 
    from the marziehf/IdiomTranslationDS dataset layout to yield enriched 
    semantic chunks for RAG pipelines.
    
    :param repo_dir: Path to the 'de-en' directory of the cloned repository
    :param split: Which data split to target ('train' or 'test')
    """
    de_path = os.path.join(repo_dir, f"{split}.de")
    en_path = os.path.join(repo_dir, f"{split}.en")
    info_path = os.path.join(repo_dir, f"{split}.info")
    
    # Verify all files exist before processing
    for file_path in [de_path, en_path, info_path]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing required dataset file: {file_path}")
            
    # Open all three files concurrently to process line-by-line
    with open(de_path, 'r', encoding='utf-8') as de_f, \
         open(en_path, 'r', encoding='utf-8') as en_f, \
         open(info_path, 'r', encoding='utf-8') as info_f:
             
        for line_num, (de_line, en_line, info_line) in enumerate(zip(de_f, en_f, info_f), start=1):
            de_sentence = de_line.strip()
            en_sentence = en_line.strip()
            info_data = info_line.strip().split('\t')
            
            # Skip empty lines or malformed info rows
            if not de_sentence or not en_sentence or len(info_data) < 3:
                continue
                
            # Extract structured metadata from the tab-separated .info row
            # Format: [Alignment Index] \t [German Base Form] \t [English Base Idiom]
            _alignment_idx = info_data[0]
            german_base = info_data[1].strip()
            english_base = info_data[2].strip()
            
            # Construct the semantic chunk template perfectly
            semantic_chunk = (
                f"Dataset: German Umgangssprache\n"
                f"Target Idiom (Base Form): {german_base}\n"
                f"English Meaning: {english_base}\n"
                f"German Contextual Example: {de_sentence}\n"
                f"English Translation: {en_sentence}"
            )
            
            # Return both the text chunk for embedding and clean metadata for your DB
            metadata = {
                "line": line_num,
                "german_base": german_base,
                "english_base": english_base,
                "raw_german": de_sentence,
                "raw_english": en_sentence
            }
            
            yield semantic_chunk, metadata

# --- Implementation Example ---
if __name__ == "__main__":
    # Point this to your local directory containing train.de, train.en, train.info
    DATASET_DIRECTORY = "./IdiomDataset/de-en"
    
    try:
        # Pull the first 3 chunks as a verification test
        chunk_generator = generate_semantic_chunks(DATASET_DIRECTORY, split="idiom_trainplus")
        
        print("🚀 Successfully generated semantic chunks:\n")
        for i, (chunk, meta) in enumerate(chunk_generator):
            if i >= 3: 
                break
                
            print(f"=== [CHUNK #{meta['line']}] ===")
            print(chunk)
            print("-" * 40)
        
        save_chunks_to_jsonl(chunk_generator)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
       