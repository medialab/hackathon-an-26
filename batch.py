from openai import OpenAI
import argparse
import uuid
import os
import casanova
import json
import glob
import sys
import jsonl
import tqdm

client = OpenAI()

EMBED_TEMP = {
    "custom_id": None,
    "method": "POST",
    "url": "/v1/embeddings",
    "body": {"input": None, "model": "text-embedding-3-small"},
}


def _create_jsonl(texts):
    lines = ""
    id = None
    for text in texts:
        id = f"request-{str(uuid.uuid1())[:8]}"
        d = EMBED_TEMP.copy()
        d["custom_id"] = id
        d["body"]["input"] = text

        lines += json.dumps(d, ensure_ascii=False) + "\n"

    return lines, id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    upload_parser = subparsers.add_parser(
        "prepare", help="upload a batch from a CSV file"
    )
    upload_parser.add_argument("csv_path", help="Path to input CSV file")
    upload_parser.add_argument("column_name", help="Column name to process")

    upload_parser = subparsers.add_parser("send", help="upload a batch from a CSV file")
    upload_parser.add_argument("csv_path", help="Path to input CSV file")
    upload_parser.add_argument("column_name", help="Column name to process")

    upload_parser = subparsers.add_parser(
        "upload", help="upload a batch from a CSV file"
    )
    upload_parser.add_argument("jsonl", nargs="*", help="Path to jsonl files")

    info_parser = subparsers.add_parser("info", help="Get batch information")
    info_parser.add_argument("batch_id", help="Batch ID")

    download_parser = subparsers.add_parser("download", help="Download batch results")
    download_parser.add_argument("input_csv_path", help="Path to input CSV file")
    download_parser.add_argument(
        "column_name", help="Column name to store the embedding"
    )

    download_parser = subparsers.add_parser("enrich", help="Download batch results")
    download_parser.add_argument("input_csv_path", help="Path to input CSV file")
    download_parser.add_argument(
        "column_name", help="Column name to store the embedding"
    )
    download_parser.add_argument(
        "output_dir", help="Path to output dir of embeddings files"
    )
    download_parser.add_argument(
        "request_dir", help="Path to request dir of embeddings files"
    )

    return parser


def prepare(path, column, limit=500):

    name = path.split("/")[-1].replace(".", "_")

    os.makedirs("requests", exist_ok=True)
    with casanova.reader(path) as reader:
        h = reader.headers[column]

        acc = []
        for i, row in enumerate(reader):
            acc.append(row[h])

            if i % limit == 0 and i > 0:
                lines, id = _create_jsonl(acc)
                with open(f"requests/{name}_{i}.jsonl", "w") as export:
                    export.write(lines)
                acc = []

        lines, id = _create_jsonl(acc)
        with open(f"requests/{name}_{i}.jsonl", "w") as export:
            export.write(lines)
        acc = []


def upload(jsonl):
    print("input_ids")
    for file in jsonl:
        input_file = client.files.create(file=open(file, "rb"), purpose="batch")
        print(input_file.id)


def send(path, column):
    print("batch_ids")
    with casanova.reader(path) as reader:
        c = reader.headers[column]
        for i, row in enumerate(reader):
            id = row[c]
            res = client.batches.create(
                input_file_id=id,
                endpoint="/v1/embeddings",
                completion_window="24h",
                metadata={"description": f"job {i}"},
            )
            print(res.id)


def download(path, column):
    os.makedirs("outputs", exist_ok=True)
    with casanova.reader(path) as reader:
        c = reader.headers[column]
        for i, row in enumerate(reader):
            id = row[c]
            batch = client.batches.retrieve(id)
            res = client.files.content(batch.output_file_id)
            with open(f"outputs/output_{i}.csv", "w") as export:
                export.write(res.text)


def enrich(origin, column, dir_output, dir_request):
    files_output = glob.glob(f"{dir_output}/*")
    files_request = glob.glob(f"{dir_request}/*")

    files_output = [
        (int(f.replace(f"{dir_output}/output_", "").replace(".csv", "")), f)
        for f in files_output
    ]
    files_request = [
        (int(f.replace(f"{dir_request}/facebook_csv_", "").replace(".jsonl", "")), f)
        for f in files_request
    ]
    files_output = sorted(files_output, key=lambda a: a[0])
    files_request = sorted(files_request, key=lambda a: a[0])
    
    #print([a for a, _ in files_request], files_output)

    jsonl_items = {}
    for _, f in tqdm.tqdm(files_output):
        items = jsonl.load(f)
        for it in items:
            c_id = it["custom_id"]
            embedding = it["response"]["body"]["data"][0]["embedding"]
            jsonl_items[c_id] = embedding

    embeddings = []
    for _, f in tqdm.tqdm(files_request):
        items = jsonl.load(f)
        for it in items:
            c_id = it["custom_id"]
            embedding = jsonl_items[c_id]
            embeddings.append(embedding)

    with casanova.enricher(origin, sys.stdout, add=["embedding_text"]) as enricher:
        for row, embed in zip(enricher, embeddings):
            enricher.writerow(row, add=[embed])


def info():
    pass


if __name__ == "__main__":
    parser = build_parser()

    args = parser.parse_args()

    if args.command == "prepare":
        prepare(args.csv_path, args.column_name)
    elif args.command == "upload":
        upload(args.jsonl)
    elif args.command == "send":
        send(args.csv_path, args.column_name)
    elif args.command == "info":
        info(args.batch_id)
    elif args.command == "enrich":
        enrich(args.input_csv_path, args.column_name, args.output_dir, args.request_dir)
    elif args.command == "download":
        download(args.input_csv_path, args.column_name)
