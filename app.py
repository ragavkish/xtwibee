import json
import logging
import os
import signal
from tweepy import StreamingClient, StreamRule
from hdfs import InsecureClient

HDFS_OUTPUT_PATH = "/tweets/streamed_tweets.json"
HDFS_URL = "http://localhost:50070"
BATCH_SIZE = 100

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("streaming.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TwitterStreamToHDFS")

class HDFSStreamClient(StreamingClient):
    def __init__(self, bearer_token, hdfs_client, hdfs_path, batch_size=BATCH_SIZE):
        super().__init__(bearer_token)
        self.hdfs_client = hdfs_client
        self.hdfs_path = hdfs_path
        self.buffer = []
        self.batch_size = batch_size

    def on_data(self, raw_data):
        try:
            tweet = json.loads(raw_data)
            if "data" in tweet:
                self.buffer.append(tweet["data"])
                if len(self.buffer) >= self.batch_size:
                    self.flush_to_hdfs()
        except Exception as e:
            logger.error(f"Error processing tweet: {e}")

    def flush_to_hdfs(self):
        try:
            logger.info(f"Flushing {len(self.buffer)} tweets to HDFS...")
            with self.hdfs_client.write(self.hdfs_path, append=True, encoding="utf-8") as hdfs_file:
                for tweet in self.buffer:
                    hdfs_file.write(json.dumps(tweet) + "\n")
            self.buffer = []
        except Exception as e:
            logger.error(f"Error writing to HDFS: {e}")

    def on_errors(self, errors):
        logger.error(f"Streaming error: {errors}")

    def on_connection_error(self):
        logger.error("Connection error!")
        self.disconnect()

def shutdown_handler(signum, frame):
    logger.info("Flushing data before exiting...")
    client.flush_to_hdfs()
    client.disconnect()
    exit(0)

if __name__ == "__main__":
    try:
        signal.signal(signal.SIGINT, shutdown_handler)
        BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")

        hdfs_client = InsecureClient(HDFS_URL)
        client = HDFSStreamClient(BEARER_TOKEN, hdfs_client, HDFS_OUTPUT_PATH)

        logger.info("Adding streaming rules...")

        space_keywords = (
            "space OR NASA OR astronomy OR cosmos OR rocket OR planet OR universe OR spacex "
            "OR stars OR galaxy OR black hole OR astrophysics OR exoplanet OR satellite "
            "OR orbital mechanics OR space telescope OR Mars OR Jupiter OR Milky Way OR ESA "
            "OR cosmology OR gravity OR space science OR lunar OR solar system OR asteroid "
            "OR comet OR spacecraft OR mission to Mars OR star formation OR space station "
            "OR ISS OR space exploration OR astronaut OR rocket launch OR alien life "
            "OR deep space OR Hubble OR James Webb OR JWST OR space debris OR interstellar "
            "OR dark matter OR dark energy OR cosmic rays OR Kuiper Belt OR Oort Cloud "
            "OR planetary science OR space weather"
        )

        existing_rules = client.get_rules().data
        if existing_rules:
            rule_ids = [rule.id for rule in existing_rules]
            client.delete_rules(rule_ids)

        client.add_rules([StreamRule(value=space_keywords, tag="Space and Astronomy")])
        
        logger.info("Starting Twitter stream...")
        client.filter(
            tweet_fields=["created_at", "author_id", "text"],
            expansions=["author_id"],
            languages=["en"]
        )
    except Exception as e:
        logger.error(f"Error in streaming: {e}")
