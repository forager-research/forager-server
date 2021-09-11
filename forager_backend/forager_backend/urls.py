"""forager_backend URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from forager_backend_api import views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/start_cluster", views.start_cluster, name="start_cluster"),
    path(
        "api/cluster/<slug:cluster_id>",
        views.get_cluster_status,
        name="get_cluster_status",
    ),
    path("api/stop_cluster/<slug:cluster_id>", views.stop_cluster, name="stop_cluster"),
    path("api/get_results/<slug:dataset_name>", views.get_results, name="get_results"),
    path("api/keep_alive", views.keep_alive, name="keep_alive"),
    path("api/generate_embedding", views.generate_embedding, name="generate_embedding"),
    path(
        "api/generate_text_embedding",
        views.generate_text_embedding,
        name="generate_text_embedding",
    ),
    path("api/query_knn/<slug:dataset_name>", views.query_knn, name="query_knn"),
    path("api/train_svm/<slug:dataset_name>", views.train_svm, name="train_svm"),
    path("api/query_svm/<slug:dataset_name>", views.query_svm, name="query_svm"),
    path(
        "api/query_ranking/<slug:dataset_name>",
        views.query_ranking,
        name="query_ranking",
    ),
    path(
        "api/query_images/<slug:dataset_name>", views.query_images, name="query_images"
    ),
    path(
        "api/query_metrics/<slug:dataset_name>",
        views.query_metrics,
        name="query_metrics",
    ),
    path(
        "api/query_active_validation/<slug:dataset_name>",
        views.query_active_validation,
        name="query_active_validation",
    ),
    path(
        "api/add_val_annotations", views.add_val_annotations, name="add_val_annotations"
    ),
    path("api/get_datasets", views.get_datasets, name="get_datasets"),
    path(
        "api/get_dataset_info/<slug:dataset_name>",
        views.get_dataset_info,
        name="get_dataset_info",
    ),
    path(
        "api/add_model_output/<slug:dataset_name>",
        views.add_model_output,
        name="add_model_output",
    ),
    path(
        "api/get_model_outputs/<slug:dataset_name>",
        views.get_model_outputs,
        name="get_model_outputs",
    ),
    path(
        "api/delete_model_output/<slug:model_output_id>",
        views.delete_model_output,
        name="delete_model_output",
    ),
    path("api/get_models/<slug:dataset_name>", views.get_models, name="get_models"),
    path("api/update_model", views.update_model, name="update_model"),
    path("api/delete_model", views.delete_model, name="delete_model"),
    path(
        "api/get_annotations",
        views.get_annotations,
        name="get_annotations",
    ),
    path("api/add_annotations", views.add_annotations, name="add_annotations"),
    path(
        "api/add_annotations_multi",
        views.add_annotations_multi,
        name="add_annotations_multi",
    ),
    path(
        "api/add_annotations_by_internal_identifiers/<slug:dataset_name>",
        views.add_annotations_by_internal_identifiers,
        name="add_annotations_by_internal_identifiers",
    ),
    path(
        "api/add_annotations_to_result_set",
        views.add_annotations_to_result_set,
        name="add_annotations_to_result_set",
    ),
    path("api/delete_category", views.delete_category, name="delete_category"),
    path("api/update_category", views.update_category, name="update_category"),
    path(
        "api/get_category_counts/<slug:dataset_name>",
        views.get_category_counts,
        name="get_category_counts",
    ),
    path(
        "api/train_model/<slug:dataset_name>", views.create_model, name="create_model"
    ),
    path("api/model/<slug:model_id>", views.get_model_status, name="get_model_status"),
    path(
        "api/model_inference/<slug:dataset_name>",
        views.run_model_inference,
        name="run_model_inference",
    ),
    path(
        "api/model_inference_status/<slug:job_id>",
        views.get_model_inference_status,
        name="get_model_inference_status",
    ),
    path(
        "api/stop_model_inference/<slug:job_id>",
        views.stop_model_inference,
        name="stop_model_inference",
    ),
    path("api/create_dataset", views.create_dataset, name="create_dataset"),
    path("api/delete_dataset", views.delete_dataset, name="delete_dataset"),
]
