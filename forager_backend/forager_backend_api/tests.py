import os
from unittest import mock

import responses
from django.conf import settings
from django.test import TestCase
from django.urls import reverse


def create_dataset():
    pass


class APIClientTests(TestCase):
    @responses.activate
    def test_create_dataset(self, get_mock, post_mock):
        responses.add(
            responses.POST,
            settings.EMBEDDING_SERVER_ADDRESS + "/start_embedding_job",
            json={"job_id": "0"},
            status=200,
        )
        responses.add(
            responses.POST,
            settings.EMBEDDING_SERVER_ADDRESS + "/start_embedding_job",
            json={"job_id": "1"},
            status=200,
        )

        # Wait for one job
        responses.add(
            responses.POST,
            settings.EMBEDDING_SERVER_ADDRESS + "/embedding_job_status",
            json={"has_job": True, "finished": False, "failed": False},
            status=200,
        )

        # Job completed
        responses.add(
            responses.POST,
            settings.EMBEDDING_SERVER_ADDRESS + "/embedding_job_status",
            json={
                "has_job": True,
                "finished": True,
                "failed": False,
                "image_list_path": "/dummy/path1",
                "embeddings_path": "/dummy/path1",
            },
            status=200,
        )
        responses.add(
            responses.POST,
            settings.EMBEDDING_SERVER_ADDRESS + "/embedding_job_status",
            json={
                "has_job": True,
                "finished": True,
                "failed": False,
                "image_list_path": "/dummy/path2",
                "embeddings_path": "/dummy/path2",
            },
            status=200,
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
            os.makedirs(os.path.join(tmpdirname, "train"))
            os.makedirs(os.path.join(tmpdirname, "val"))
            with open(os.path.join(tmpdirname, "train", "image1.jpg"), "wb") as f:
                f.write(b"0x0")
            with open(os.path.join(tmpdirname, "train", "image2.jpg"), "wb") as f:
                f.write(b"0x0")
            with open(os.path.join(tmpdirname, "val", "image2.jpg"), "wb") as f:
                f.write(b"0x0")
            params = {
                "dataset": "test",
                "train_images_directory": os.path.join(tmpdirname, "train"),
                "val_images_directory": os.path.join(tmpdirname, "val"),
            }

            response = self.client.post(
                reverse("forager_backend_api:create_dataset"),
                params,
                content_type="application/json",
            )

        get_mock.assert_called()
        post_mock.assert_called()

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No polls are available.")
        self.assertQuerysetEqual(response.context["latest_question_list"], [])

    def test_import_labels(self):
        """
        If no questions exist, an appropriate message is displayed.
        """
        response = self.client.get(reverse("polls:index"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No polls are available.")
        self.assertQuerysetEqual(response.context["latest_question_list"], [])

    def test_export_labels(self):
        """
        If no questions exist, an appropriate message is displayed.
        """
        response = self.client.get(reverse("polls:index"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No polls are available.")
        self.assertQuerysetEqual(response.context["latest_question_list"], [])

    def test_import_embeddings(self):
        """
        If no questions exist, an appropriate message is displayed.
        """
        response = self.client.get(reverse("polls:index"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No polls are available.")
        self.assertQuerysetEqual(response.context["latest_question_list"], [])

    def test_import_scores(self):
        """
        If no questions exist, an appropriate message is displayed.
        """
        response = self.client.get(reverse("polls:index"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No polls are available.")
        self.assertQuerysetEqual(response.context["latest_question_list"], [])
