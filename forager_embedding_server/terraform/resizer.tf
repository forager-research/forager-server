variable "resizer_image_name" {
  type    = string
  default = "gcr.io/visualdb-1046/forager-index-resizer:latest"
}

variable "resizer_node_pool_name" {
  type    = string
  default = "resizer-np"
}

variable "resizer_num_nodes" {
  type    = number
  default = 1
}

variable "resizer_node_type" {
  type    = string
  default = "n2-standard-4"
}

variable "resizer_nproc" {
  type    = number
  default = 4
}

locals {
  resizer_external_port = 5000
  resizer_internal_port = 5000
  resizer_app_name      = "resizer"
  resizer_disk_size_gb  = 10
}

resource "google_container_node_pool" "resizer_np" {
  count      = var.create_node_pools_separately ? 1 : 0
  name       = var.resizer_node_pool_name
  location   = var.zone
  cluster    = google_container_cluster.cluster.name
  node_count = var.resizer_num_nodes

  node_config {
    preemptible  = true
    machine_type = var.resizer_node_type
    disk_size_gb = local.trainer_disk_size_gb
    oauth_scopes = local.node_pool_oauth_scopes
  }

  depends_on = [kubernetes_persistent_volume_claim.nfs_claim]
}

resource "kubernetes_deployment" "resizer_dep" {
  metadata {
    name = "resizer-dep"
    labels = {
      app = local.resizer_app_name
    }
  }

  spec {
    replicas = var.resizer_num_nodes

    selector {
      match_labels = {
        app = local.resizer_app_name
      }
    }

    template {
      metadata {
        labels = {
          app = local.resizer_app_name
        }
      }

      spec {
        container {
          image = var.resizer_image_name
          name  = local.resizer_app_name

          env {
            name  = "PORT"
            value = local.resizer_internal_port
          }

          env {
            name = "NPROC"
            value = var.resizer_nproc
          }

          port {
            container_port = local.resizer_internal_port
          }

          volume_mount {
            mount_path = local.nfs_mount_dir
            name       = local.nfs_volume_name
          }

          volume_mount {
            mount_path = "/dev/shm"
            name       = "dshm"
          }
        }

        affinity {
          pod_anti_affinity {
            required_during_scheduling_ignored_during_execution {
              label_selector {
                match_expressions {
                  key      = "app"
                  operator = "In"
                  values   = [local.resizer_app_name]
                }
              }

              topology_key = "kubernetes.io/hostname"
            }
          }
        }

        volume {
          name = local.nfs_volume_name

          persistent_volume_claim {
            claim_name = kubernetes_persistent_volume_claim.nfs_claim.metadata.0.name
          }
        }

        volume {
          name = "dshm"

          empty_dir {
            medium = "Memory"
          }
        }

        node_selector = {
          "cloud.google.com/gke-nodepool" = var.resizer_node_pool_name
        }
      }
    }
  }

  depends_on = [google_container_cluster.cluster, google_container_node_pool.resizer_np]
}

resource "kubernetes_service" "resizer_svc" {
  metadata {
    name = "${kubernetes_deployment.resizer_dep.metadata.0.name}-svc"
  }
  spec {
    selector = kubernetes_deployment.resizer_dep.metadata.0.labels
    port {
      port        = local.resizer_external_port
      target_port = local.resizer_internal_port
    }

    type = "LoadBalancer"
  }
}

output "resizer_url" {
  value = "http://${kubernetes_service.resizer_svc.status.0.load_balancer.0.ingress.0.ip}:${local.resizer_external_port}"
}

output "num_resizers" {
  value = var.resizer_num_nodes
}

output "resizer_nproc" {
  value = var.resizer_nproc
}
