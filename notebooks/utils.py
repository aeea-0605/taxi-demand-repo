import configparser

config = configparser.ConfigParser()
config.read('./data.ini')

info = config['gcs']