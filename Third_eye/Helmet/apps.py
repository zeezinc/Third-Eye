from django.apps import AppConfig
import html
import pathlib
import os

class HelmetConfig(AppConfig):
    name = 'Helmet'
    MODEL_PATH = Path("model")
    