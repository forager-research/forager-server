variable "bgsplit_trainer_image_name" {
  type    = string
  default = "gcr.io/visualdb-1046/forager-bgsplit-trainer"
}

variable "bgsplit_trainer_node_pool_name" {
  type    = string
  default = "bgsplit-trainer-np"
}

variable "bgsplit_trainer_num_nodes" {
  type    = number
  default = 1
}

variable "bgsplit_trainer_node_type" {
  type    = string
  default = "n1-standard-32"
}

variable "bgsplit_trainer_accelerator_type" {
  type    = string
  default = "nvidia-tesla-v100"
}

variable "bgsplit_trainer_accelerator_count" {
  type    = number
  default = 4
}

variable "bgsplit_trainer_gpus" {
  type    = number
  default = 4
}

locals {
  bgsplit_trainer_external_port = 5000
  bgsplit_trainer_internal_port = 5000
  bgsplit_trainer_app_name      = "bgsplit-trainer"
  bgsplit_trainer_disk_size_gb  = 20
}

resource "google_container_node_pool" "bgsplit_trainer_np" {
  count      = var.create_node_pools_separately ? 1 : 0
  name       = var.bgsplit_trainer_node_pool_name
  location   = var.zone
  cluster    = google_container_cluster.cluster.name
  node_count = var.bgsplit_trainer_num_nodes

  node_config {
    preemptible  = true
    machine_type = var.bgsplit_trainer_node_type
    disk_size_gb = local.bgsplit_trainer_disk_size_gb
    oauth_scopes = local.node_pool_oauth_scopes

    guest_accelerator {
      type  = var.bgsplit_trainer_accelerator_type
      count = var.bgsplit_trainer_accelerator_count
    }
  }

  depends_on = [kubernetes_persistent_volume_claim.nfs_claim]
}

resource "kubectl_manifest" "bgsplit_trainer_dep" {
  count = var.bgsplit_trainer_num_nodes

  yaml_body = <<YAML
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bgsplit-trainer-dep-${count.index}
  labels:
    app: ${local.bgsplit_trainer_app_name}-${count.index}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ${local.bgsplit_trainer_app_name}-${count.index}
  template:
    metadata:
      labels:
        app: ${local.bgsplit_trainer_app_name}-${count.index}
    spec:
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - key: app
                operator: In
                values:
                - ${local.bgsplit_trainer_app_name}
            topologyKey: kubernetes.io/hostname
      containers:
      - image: ${var.bgsplit_trainer_image_name}
        name: ${local.bgsplit_trainer_app_name}
        resources:
          limits:
            cpu: "28"
            memory: "100G"
            nvidia.com/gpu: ${var.bgsplit_trainer_gpus}
          requests:
            cpu: "28"
            memory: "100G"
        env:
        - name: PORT
          value: "${local.bgsplit_trainer_internal_port}"
        - name: PYTHONUNBUFFERED
          value: "1"
        - name: PYTHONIOENCODING
          value: "UTF-8"
        ports:
        - containerPort: ${local.bgsplit_trainer_internal_port}
        volumeMounts:
        - mountPath: ${local.nfs_mount_dir}
          name: ${local.nfs_volume_name}
      nodeSelector:
        cloud.google.com/gke-nodepool: ${var.bgsplit_trainer_node_pool_name}
      tolerations:
      - effect: NoSchedule
        key: nvidia.com/gpu
        operator: Exists
      volumes:
      - name: ${local.nfs_volume_name}
        persistentVolumeClaim:
          claimName: ${kubernetes_persistent_volume_claim.nfs_claim.metadata.0.name}
      hostIPC: true

YAML

  depends_on = [google_container_cluster.cluster, google_container_node_pool.bgsplit_trainer_np]
}

resource "kubernetes_service" "bgsplit_trainer_svc" {
  count = var.bgsplit_trainer_num_nodes

  metadata {
    name = "bgsplit-trainer-dep-${count.index}-svc"
  }
  spec {
    selector = {
      app = "${local.bgsplit_trainer_app_name}-${count.index}"
    }
    port {
      port        = local.bgsplit_trainer_external_port
      target_port = local.bgsplit_trainer_internal_port
    }

    type = "LoadBalancer"
  }
}

output "bgsplit_trainer_urls" {
  value = [
    for svc in kubernetes_service.bgsplit_trainer_svc:
    "http://${svc.status.0.load_balancer.0.ingress.0.ip}:${local.bgsplit_trainer_external_port}"
  ]
}
