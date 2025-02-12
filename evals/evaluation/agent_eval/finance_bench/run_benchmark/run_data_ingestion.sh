
output=$WORKDIR/datasets/financebench/dataprep/

python ingest_data.py \
--output $output \
--ingest_option docling \
--retriever_option plain \