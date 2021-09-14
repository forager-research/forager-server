variable "disk_gb" {
  type    = number
  default = 2560
}

locals {
  nfs_mount_dir   = "/shared"
  nfs_volume_name = "nfs"
}

resource "google_filestore_instance" "nfs" {
  name = "nfs-${random_uuid.cluster_id.result}"
  zone = var.zone
  tier = "BASIC_SSD"

  file_shares {
    capacity_gb = var.disk_gb
    name        = "share"
  }

  networks {
    network = var.network
    modes   = ["MODE_IPV4"]
  }
}

resource "kubernetes_storage_class" "nfs_class" {
  metadata {
    name = "nfs"
  }
  storage_provisioner = "kubernetes.io/no-provisioner"
}

resource "kubernetes_persistent_volume" "nfs_volume" {
  metadata {
    name = "nfs-volume"
  }
  spec {
    capacity = {
      storage = "${var.disk_gb}Gi"
    }
    access_modes = ["ReadWriteMany"]
    persistent_volume_source {
      nfs {
        path = "/${google_filestore_instance.nfs.file_shares.0.name}"
        server = google_filestore_instance.nfs.networks.0.ip_addresses.0
      }
    }
    storage_class_name = kubernetes_storage_class.nfs_class.metadata.0.name
  }
}

resource "kubernetes_persistent_volume_claim" "nfs_claim" {
  metadata {
    name = "nfs-claim"
  }
  spec {
    access_modes = ["ReadWriteMany"]
    resources {
      requests = {
        storage = "${var.disk_gb}Gi"
      }
    }
    volume_name = kubernetes_persistent_volume.nfs_volume.metadata.0.name
    storage_class_name = kubernetes_storage_class.nfs_class.metadata.0.name
  }
}

output "nfs_mount_dir" {
  value = local.nfs_mount_dir
}

output "nfs_url" {
  value = "${google_filestore_instance.nfs.networks.0.ip_addresses.0}:/${google_filestore_instance.nfs.file_shares.0.name}"
}
