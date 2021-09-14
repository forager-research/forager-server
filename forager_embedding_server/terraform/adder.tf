variable "adder_image_name" {
  type    = string
  default = "gcr.io/visualdb-1046/forager-index-adder:latest"
}

variable "adder_node_pool_name" {
  type    = string
  default = "adder-np"
}

variable "adder_num_nodes" {
  type    = number
  default = 1
}

variable "adder_node_type" {
  type    = string
  default = "n2-standard-2"
}

variable "adder_nproc" {
  type    = number
  default = 1
}

locals {
  adder_external_port = 5000
  adder_internal_port = 5000
  adder_app_name      = "adder"
  adder_disk_size_gb  = 10
}

resource "google_container_node_pool" "adder_np" {
  count      = var.create_node_pools_separately ? 1 : 0
  name       = var.adder_node_pool_name
  location   = var.zone
  cluster    = google_container_cluster.cluster.name
  node_count = var.adder_num_nodes

  node_config {
    preemptible  = true
    machine_type = var.adder_node_type
    disk_size_gb = local.adder_disk_size_gb
    oauth_scopes = local.node_pool_oauth_scopes
  }

  depends_on = [kubernetes_persistent_volume_claim.nfs_claim]
}

resource "kubernetes_deployment" "adder_dep" {
  metadata {
    name = "adder-dep"
    labels = {
      app = local.adder_app_name
    }
  }

  spec {
    replicas = var.adder_num_nodes

    selector {
      match_labels = {
        app = local.adder_app_name
      }
    }

    template {
      metadata {
        labels = {
          app = local.adder_app_name
        }
      }

      spec {
        container {
          image = var.adder_image_name
          name  = local.adder_app_name

          env {
            name  = "PORT"
            value = local.adder_internal_port
          }

          env {
            name = "NPROC"
            value = var.adder_nproc
          }

          port {
            container_port = local.adder_internal_port
          }

          volume_mount {
            mount_path = local.nfs_mount_dir
            name       = local.nfs_volume_name
          }
        }

        affinity {
          pod_anti_affinity {
            required_during_scheduling_ignored_during_execution {
              label_selector {
                match_expressions {
                  key      = "app"
                  operator = "In"
                  values   = [local.adder_app_name]
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

        node_selector = {
          "cloud.google.com/gke-nodepool" = var.adder_node_pool_name
        }
      }
    }
  }

  depends_on = [google_container_cluster.cluster, google_container_node_pool.adder_np]
}

resource "kubernetes_service" "adder_svc" {
  metadata {
    name = "${kubernetes_deployment.adder_dep.metadata.0.name}-svc"
  }
  spec {
    selector = kubernetes_deployment.adder_dep.metadata.0.labels
    port {
      port        = local.adder_external_port
      target_port = local.adder_internal_port
    }

    type = "LoadBalancer"
  }
}

output "adder_url" {
  value = "http://${kubernetes_service.adder_svc.status.0.load_balancer.0.ingress.0.ip}:${local.adder_external_port}"
}

output "num_adders" {
  value = var.adder_num_nodes
}

output "adder_nproc" {
  value = var.adder_nproc
}
