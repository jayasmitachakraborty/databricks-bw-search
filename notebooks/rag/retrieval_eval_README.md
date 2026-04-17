# Retrieval evaluation

Run:
python evaluate_retrieval.py --gold-csv gold_set_template.csv -k 10 --output-json results.json --output-predictions-csv predictions.csv

Gold CSV columns:
- query_id
- query_text
- relevant_chunk_id
- relevance_label (0-3)
- split
- notes

Labels:
- 0 irrelevant
- 1 partially relevant
- 2 relevant
- 3 ideal
