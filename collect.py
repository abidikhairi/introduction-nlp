import os
import pandas as pd
import tweepy as tw


class ConsoleWriter(tw.Stream):
	def __init__(self, consumer_key, consumer_secret, access_token, access_token_secret, **kwargs):
		super().__init__(consumer_key, consumer_secret, access_token, access_token_secret, **kwargs)

		self.max_tweets = 2000
		self.pdata = []

	def on_status(self, status):

		tweet = {
			"id": status.id_str,
			"text": status.text
		}
		
		print(tweet)
		print('-'*100)
		
		if self.max_tweets == 0:
			pd.DataFrame(self.pdata).to_csv('./data/tweets.csv')
			self.running = False
		
		self.pdata.append(tweet)
		self.max_tweets -= 1
		return super().on_status(status)


if __name__ == '__main__':
	consumer_key = os.environ['CONSUMER_KEY']
	consumer_secret = os.environ['CONSUMER_SECRET']
	access_key = os.environ['ACCESS_TOKEN']
	access_secret = os.environ['ACCESS_TOKEN_SECRET']

	keywords = ["python", "datascience"]	
	langs = ["en"]

	auth = tw.OAuthHandler(consumer_key=consumer_key, consumer_secret=consumer_secret)
	auth.set_access_token(key=access_key, secret=access_secret)


	stream = ConsoleWriter(consumer_key, consumer_secret, access_key, access_secret, max_retries=10)
	stream.filter(track=keywords, languages=langs)
