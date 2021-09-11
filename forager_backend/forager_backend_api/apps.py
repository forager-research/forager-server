from django.apps import AppConfig


class ForagerServerApiConfig(AppConfig):
    name = 'forager_backend_api'

    def ready(self):
        import forager_backend_api.signals
