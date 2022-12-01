# Cannot parse youtube search page through simple 'requests' moudle
from requests_html import HTMLSession
from pytube import YouTube
import os
import subprocess

#import sys
#sys.path.append('../demucs/demucs')
#from separate import main as demucs

class YouTubeMusicDownloader(object):
	def __init__(self, song_title, artist):
		self.song_title = song_title
		self.artist = artist

	def download(self):
		search_url = "https://www.youtube.com/results?search_query="+ \
						self.song_title+"+"+self.artist+"+가사"
		search_url = search_url.replace(" ", "+")

		new_file = self.song_title + ' - ' + self.artist + '.mp3'
		if os.path.exists(new_file):
			print(new_file, "already exists, skip downloading...")
		else:
			print("HTML Session Starts")
			session = HTMLSession()
			response = session.get(search_url)
			response.html.render()

			links = response.html.find('a#video-title')
			if not links:
				print("No Search Result")
				return None

			link = links[0] # set object
			link = list(link.absolute_links)[0]

			print("Downloading mp3 file...")
			yt = YouTube(link)
			video = yt.streams.filter(only_audio=True).first()

			destination = "."
			out_file = video.download(output_path=destination)

			# base, ext = os.path.splitext(out_file)
			os.rename(out_file, new_file)

			print("Successfully downloaded", new_file)

		return new_file

class AudioSeparator(object):
	INSTRUMENTS = ['vocals', 'drums', 'bass', 'other']
	AUDIO_DIR = './'
	def __init__(self, audio_fname):
		self.audio_fname = self.AUDIO_DIR + audio_fname

	def separate(self, target_inst):
		if target_inst not in self.INSTRUMENTS:
			assert False, str(target_inst) + " not in " + str(self.INSTRUMENTS)
		
		args = ['demucs', '--two-stems='+target_inst, 
				'--filename', './separated/{stem}.{ext}', self.audio_fname]
		subprocess.run(args, shell=True, check=True)

		"""
		if not len(sys.argv) == 1:
			assert False, str(sys.argv)
		sys.argv.extend(['-n', 'mdx_q',
						 '--two-stems='+target_inst, 
						 '--filename', './separated/{stem}.{ext}', self.audio_fname])

		demucs()
		"""






if __name__ == '__main__':
	song_title = input("Title: ")
	artist = input("Artist: ")

	downloader = YouTubeMusicDownloader(song_title, artist)
	audiofile = downloader.download()
	separator = AudioSeparator(audiofile)
	separator.separate('vocals')


