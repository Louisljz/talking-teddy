import datetime
from google.cloud import storage
from google.oauth2.service_account import Credentials
from supabase import create_client
import os

class BucketUtils:
    def __init__(self):
        credentials = Credentials.from_service_account_file('talking-teddy-service.json')
        self.client = storage.Client(credentials=credentials)
        self.supabase = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))

    def upload_blob(self, source_file_name, destination_blob_name):
        bucket_name = "talking_teddy"
        bucket = self.client.get_bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        # Upload the file
        blob.upload_from_filename(source_file_name)

        # Generate signed URL (valid for 1 hour)
        url = blob.generate_signed_url(expiration=datetime.timedelta(hours=1))
        print(f"File {source_file_name} uploaded to {destination_blob_name}.")
        
        
        snapshotType = 0
        
        if destination_blob_name.startswith("snapshots/videos"):
            snapshotType = 1
        
        self.supabase.table("snapshot").update({
            "link": url,
            "type": snapshotType
        }).eq("id", 1).execute()
        
    