variable "bgsplit_mapper_image_name" {
  type    = string
  default = "gcr.io/visualdb-1046/forager-bgsplit-mapper:latest"
}

variable "bgsplit_mapper_node_pool_name" {
  type    = string
  default = "bgsplit-mapper-np"
}

variable "bgsplit_mapper_num_nodes" {
  type    = number
  default = 40
}

variable "bgsplit_mapper_node_type" {
  type    = string
  default = "n2-standard-16"
}

variable "bgsplit_mapper_nproc" {
  type    = number
  default = 1
}

variable "bgsplit_mapper_gpus" {
  type    = number
  default = 0
}

locals {
  bgsplit_mapper_external_port = 5000
  bgsplit_mapper_internal_port = 5000
  bgsplit_mapper_app_name      = "bgsplit-mapper"
  bgsplit_mapper_disk_size_gb  = 30
}

resource "google_container_node_pool" "bgsplit_mapper_np" {
  count      = var.create_node_pools_separately ? 1 : 0
  name       = var.bgsplit_mapper_node_pool_name
  location   = var.zone
  cluster    = google_container_cluster.cluster.name
  node_count = var.bgsplit_mapper_num_nodes

  node_config {
    preemptible  = true
    machine_type = var.bgsplit_mapper_node_type
    disk_size_gb = local.bgsplit_mapper_disk_size_gb
    oauth_scopes = local.node_pool_oauth_scopes
  }

  depends_on = [kubernetes_persistent_volume_claim.nfs_claim]
}

resource "kubectl_manifest" "bgsplit_mapper_dep" {
  yaml_body = <<YAML
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bgsplit-mapper-dep
  labels:
    app: ${local.bgsplit_mapper_app_name}
spec:
  replicas: ${var.bgsplit_mapper_num_nodes}
  selector:
    matchLabels:
      app: ${local.bgsplit_mapper_app_name}
  template:
    metadata:
      labels:
        app: ${local.bgsplit_mapper_app_name}
    spec:
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - key: app
                operator: In
                values:
                - ${local.bgsplit_mapper_app_name}
            topologyKey: kubernetes.io/hostname
      containers:
      - image: ${var.bgsplit_mapper_image_name}
        name: ${local.bgsplit_mapper_app_name}
        resources:
          limits:
            cpu: "14"
            memory: "60G"
            nvidia.com/gpu: ${var.bgsplit_mapper_gpus}
          requests:
            cpu: "14"
            memory: "60G"
        env:
        - name: PORT
          value: "${local.bgsplit_mapper_internal_port}"
        - name: PYTHONUNBUFFERED
          value: "1"
        - name: PYTHONIOENCODING
          value: "UTF-8"
        - name: NPROC
          value: "${var.bgsplit_mapper_nproc}"
        ports:
        - containerPort: ${local.bgsplit_mapper_internal_port}
        volumeMounts:
        - mountPath: ${local.nfs_mount_dir}
          name: ${local.nfs_volume_name}
      nodeSelector:
        cloud.google.com/gke-nodepool: ${var.bgsplit_mapper_node_pool_name}
      volumes:
      - name: ${local.nfs_volume_name}
        persistentVolumeClaim:
          claimName: ${kubernetes_persistent_volume_claim.nfs_claim.metadata.0.name}
      hostIPC: true

YAML

  depends_on = [google_container_cluster.cluster, google_container_node_pool.bgsplit_mapper_np]
}


resource "kubernetes_service" "bgsplit_mapper_svc" {
  metadata {
    name = "bgsplit-mapper-dep-svc"
  }
  spec {
    selector = {
        app: local.bgsplit_mapper_app_name
    }
    port {
      port        = local.bgsplit_mapper_external_port
      target_port = local.bgsplit_mapper_internal_port
    }

    type = "LoadBalancer"
  }
}

output "bgsplit_mapper_url" {
  value = "http://${kubernetes_service.bgsplit_mapper_svc.status.0.load_balancer.0.ingress.0.ip}:${local.bgsplit_mapper_external_port}"
}

output "num_bgsplit_mappers" {
  value = var.bgsplit_mapper_num_nodes
}

output "bgsplit_mapper_nproc" {
  value = var.bgsplit_mapper_nproc
}
