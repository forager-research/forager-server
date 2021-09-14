variable "mapper_image_name" {
  type    = string
  default = "gcr.io/visualdb-1046/forager-index-mapper:latest"
}

variable "mapper_node_pool_name" {
  type    = string
  default = "mapper-np"
}

variable "mapper_num_nodes" {
  type    = number
  default = 1
}

variable "mapper_node_type" {
  type    = string
  default = "n2-standard-16"
}

variable "mapper_nproc" {
  type    = number
  default = 16
}

locals {
  mapper_external_port = 5000
  mapper_internal_port = 5000
  mapper_app_name      = "mapper"
  mapper_disk_size_gb  = 10
}

resource "google_container_node_pool" "mapper_np" {
  count      = var.create_node_pools_separately ? 1 : 0
  name       = var.mapper_node_pool_name
  location   = var.zone
  cluster    = google_container_cluster.cluster.name
  node_count = var.mapper_num_nodes

  node_config {
    preemptible  = true
    machine_type = var.mapper_node_type
    disk_size_gb = local.trainer_disk_size_gb
    oauth_scopes = local.node_pool_oauth_scopes
  }

  depends_on = [kubernetes_persistent_volume_claim.nfs_claim]
}

resource "kubernetes_deployment" "mapper_dep" {
  metadata {
    name = "mapper-dep"
    labels = {
      app = local.mapper_app_name
    }
  }

  spec {
    replicas = var.mapper_num_nodes

    selector {
      match_labels = {
        app = local.mapper_app_name
      }
    }

    template {
      metadata {
        labels = {
          app = local.mapper_app_name
        }
      }

      spec {
        container {
          image = var.mapper_image_name
          name  = local.mapper_app_name

          env {
            name  = "PORT"
            value = local.mapper_internal_port
          }

          env {
            name = "NPROC"
            value = var.mapper_nproc
          }

          port {
            container_port = local.mapper_internal_port
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
                  values   = [local.mapper_app_name]
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
          "cloud.google.com/gke-nodepool" = var.mapper_node_pool_name
        }
      }
    }
  }

  depends_on = [google_container_cluster.cluster, google_container_node_pool.mapper_np]
}

resource "kubernetes_service" "mapper_svc" {
  metadata {
    name = "${kubernetes_deployment.mapper_dep.metadata.0.name}-svc"
  }
  spec {
    selector = kubernetes_deployment.mapper_dep.metadata.0.labels
    port {
      port        = local.mapper_external_port
      target_port = local.mapper_internal_port
    }

    type = "LoadBalancer"
  }
}

output "mapper_url" {
  value = "http://${kubernetes_service.mapper_svc.status.0.load_balancer.0.ingress.0.ip}:${local.mapper_external_port}"
}

output "num_mappers" {
  value = var.mapper_num_nodes
}

output "mapper_nproc" {
  value = var.mapper_nproc
}
