import json
import logging
from tweepy import Stream, OAuthHandler
from tweepy.streaming import StreamListener
import hdfs



HDFS_OUTPUT_PATH = "tweets/streamed_tweets.json"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TwitterStreamToHDFS")

class HDFSStreamListener(StreamListener):
    def __init__(self, hdfs_path):
        super().__init__()
        self.hdfs_path = hdfs_path
        self.buffer = []
        self.batch_size = 100

    def on_data(self, data):
        try:
            tweet = json.loads(data)
            self.buffer.append(tweet)

            if len(self.buffer) >= self.batch_size:
                self.flush_to_hdfs()

            return True
        except Exception as e:
            logger.error(f"Error processing tweet: {e}")
            return True

    def on_error(self, status_code):
        logger.error(f"Streaming error: {status_code}")
        return status_code != 420

    def flush_to_hdfs(self):
        try:
            logger.info(f"Flushing {len(self.buffer)} tweets to HDFS...")
            with hdfs.open(self.hdfs_path, "at") as hdfs_file:
                for tweet in self.buffer:
                    hdfs_file.write(json.dumps(tweet) + "\n")
            self.buffer = []
        except Exception as e:
            logger.error(f"Error writing to HDFS: {e}")

if __name__ == "__main__":
    try:
        auth = OAuthHandler(API_KEY, API_SECRET)
        auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

        listener = HDFSStreamListener(HDFS_OUTPUT_PATH)

        logger.info("Starting Twitter stream...")
        stream = Stream(auth, listener)

        stream.filter(track=["Python", "Big Data", "Hadoop"], languages=["en"])
    except KeyboardInterrupt:
        logger.info("Streaming stopped by user.")
    except Exception as e:
        logger.error(f"Error in streaming: {e}")
