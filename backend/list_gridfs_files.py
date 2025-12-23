from pymongo import MongoClient
import gridfs
import os

# Change this if you have a different MongoDB URI
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")

def list_files():
    client = MongoClient(MONGO_URL)
    db = client["bottrainer"]
    fs = gridfs.GridFS(db)

    files_cursor = db.fs.files.find()
    print(f"Total files in GridFS: {db.fs.files.count_documents({})}\n")

    for file_doc in files_cursor:
        print(f"File ID: {file_doc['_id']}")
        print(f"Filename: {file_doc.get('filename', 'N/A')}")
        print(f"Upload Date: {file_doc.get('uploadDate', 'N/A')}")
        metadata = file_doc.get('metadata')
        print(f"Metadata: {metadata}")
        print("-" * 40)

if __name__ == "__main__":
    list_files()
