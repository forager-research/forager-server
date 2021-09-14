variable "trainer_image_name" {
  type    = string
  default = "gcr.io/visualdb-1046/forager-index-trainer"
}

variable "trainer_node_pool_name" {
  type    = string
  default = "trainer-np"
}

variable "trainer_num_nodes" {
  type    = number
  default = 1
}

variable "trainer_node_type" {
  type    = string
  default = "n1-highmem-8"
}

variable "trainer_accelerator_type" {
  type    = string
  default = "nvidia-tesla-k80"
}

variable "trainer_accelerator_count" {
  type    = number
  default = 1
}

variable "trainer_gpus" {
  type    = number
  default = 1
}

locals {
  trainer_external_port = 5000
  trainer_internal_port = 5000
  trainer_app_name      = "trainer"
  trainer_disk_size_gb  = 20
}

resource "google_container_node_pool" "trainer_np" {
  count      = var.create_node_pools_separately ? 1 : 0
  name       = var.trainer_node_pool_name
  location   = var.zone
  cluster    = google_container_cluster.cluster.name
  node_count = var.trainer_num_nodes

  node_config {
    preemptible  = true
    machine_type = var.trainer_node_type
    disk_size_gb = local.trainer_disk_size_gb
    oauth_scopes = local.node_pool_oauth_scopes

    guest_accelerator {
      type  = var.trainer_accelerator_type
      count = var.trainer_accelerator_count
    }
  }

  depends_on = [kubernetes_persistent_volume_claim.nfs_claim]
}

resource "kubectl_manifest" "trainer_dep" {
  count = var.trainer_num_nodes

  yaml_body = <<YAML
apiVersion: apps/v1
kind: Deployment
metadata:
  name: trainer-dep-${count.index}
  labels:
    app: ${local.trainer_app_name}-${count.index}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ${local.trainer_app_name}-${count.index}
  template:
    metadata:
      labels:
        app: ${local.trainer_app_name}-${count.index}
    spec:
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - key: app
                operator: In
                values:
                - ${local.trainer_app_name}
            topologyKey: kubernetes.io/hostname
      containers:
      - image: ${var.trainer_image_name}
        name: ${local.trainer_app_name}
        resources:
          limits:
            nvidia.com/gpu: ${var.trainer_gpus}
        env:
        - name: PORT
          value: "${local.trainer_internal_port}"
        ports:
        - containerPort: ${local.trainer_internal_port}
        volumeMounts:
        - mountPath: ${local.nfs_mount_dir}
          name: ${local.nfs_volume_name}
      nodeSelector:
        cloud.google.com/gke-nodepool: ${var.trainer_node_pool_name}
      tolerations:
      - effect: NoSchedule
        key: nvidia.com/gpu
        operator: Exists
      volumes:
      - name: ${local.nfs_volume_name}
        persistentVolumeClaim:
          claimName: ${kubernetes_persistent_volume_claim.nfs_claim.metadata.0.name}
YAML

  depends_on = [google_container_cluster.cluster, google_container_node_pool.trainer_np]
}

resource "kubernetes_service" "trainer_svc" {
  count = var.trainer_num_nodes

  metadata {
    name = "trainer-dep-${count.index}-svc"
  }
  spec {
    selector = {
      app = "${local.trainer_app_name}-${count.index}"
    }
    port {
      port        = local.trainer_external_port
      target_port = local.trainer_internal_port
    }

    type = "LoadBalancer"
  }
}

output "trainer_urls" {
  value = [
    for svc in kubernetes_service.trainer_svc:
    "http://${svc.status.0.load_balancer.0.ingress.0.ip}:${local.trainer_external_port}"
  ]
}
