import json
import logging
from tweepy import StreamingClient
from tweepy import StreamRule
import hdfs

HDFS_OUTPUT_PATH = "tweets/streamed_tweets.json"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TwitterStreamToHDFS")

class HDFSStreamClient(StreamingClient):
    def __init__(self, bearer_token, hdfs_path, batch_size=100):
        super().__init__(bearer_token)
        self.hdfs_path = hdfs_path
        self.buffer = []
        self.batch_size = batch_size

    def on_data(self, raw_data):
        try:
            tweet = json.loads(raw_data)
            self.buffer.append(tweet)

            if len(self.buffer) >= self.batch_size:
                self.flush_to_hdfs()
        except Exception as e:
            logger.error(f"Error processing tweet: {e}")

    def flush_to_hdfs(self):
        try:
            logger.info(f"Flushing {len(self.buffer)} tweets to HDFS...")
            with hdfs.open(self.hdfs_path, "at") as hdfs_file:
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

if __name__ == "__main__":
    try:
        BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAEE2xwEAAAAAqEZy6iHZxQ2EaHvq7rPQCikoDxs%3DtYB3diyMYd20sIlPwJQ9RosnlCKGdBdjshkq2MGqpsUoDokCZM"
        client = HDFSStreamClient(BEARER_TOKEN, HDFS_OUTPUT_PATH)

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

        try:
            client.add_rules([StreamRule(value=space_keywords, tag="Space and Astronomy")])
        except Exception as e:
            logger.error(f"Error adding rules: {e}")
            
        logger.info("Starting Twitter stream...")
        
        client.filter(
            tweet_fields=["created_at", "author_id", "text"],
            expansions=["author_id"],
            languages=["en"]
        )

    except KeyboardInterrupt:
        logger.info("Streaming stopped by user.")
    except Exception as e:
        logger.error(f"Error in streaming: {e}")
