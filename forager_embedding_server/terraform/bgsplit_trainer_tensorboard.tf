variable "bgsplit_trainer_tensorboard_image_name" {
  type    = string
  default = "tensorflow/tensorflow:2.4.1"
}

variable "bgsplit_trainer_tensorboard_node_pool_name" {
  type    = string
  default = "bgsplit-trainer-tensorboard-np"
}

variable "bgsplit_trainer_tensorboard_num_nodes" {
  type    = number
  default = 1
}

variable "bgsplit_trainer_tensorboard_node_type" {
  type    = string
  default = "n1-standard-2"
}

locals {
  bgsplit_trainer_tensorboard_external_port = 6006
  bgsplit_trainer_tensorboard_internal_port = 6006
  bgsplit_trainer_tensorboard_app_name      = "bgsplit-trainer-tensorboard"
  bgsplit_trainer_tensorboard_disk_size_gb  = 20
}

resource "google_container_node_pool" "bgsplit_trainer_tensorboard_np" {
  count      = var.create_node_pools_separately ? 1 : 0
  name       = var.bgsplit_trainer_tensorboard_node_pool_name
  location   = var.zone
  cluster    = google_container_cluster.cluster.name
  node_count = var.bgsplit_trainer_tensorboard_num_nodes

  node_config {
    preemptible  = true
    machine_type = var.bgsplit_trainer_tensorboard_node_type
    disk_size_gb = local.bgsplit_trainer_tensorboard_disk_size_gb
    oauth_scopes = local.node_pool_oauth_scopes
  }

  depends_on = [kubernetes_persistent_volume_claim.nfs_claim]
}

resource "kubectl_manifest" "bgsplit_trainer_tensorboard_dep" {
  count = var.bgsplit_trainer_tensorboard_num_nodes

  yaml_body = <<YAML
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bgsplit-trainer-tensorboard-dep-${count.index}
  labels:
    app: ${local.bgsplit_trainer_tensorboard_app_name}-${count.index}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ${local.bgsplit_trainer_tensorboard_app_name}-${count.index}
  template:
    metadata:
      labels:
        app: ${local.bgsplit_trainer_tensorboard_app_name}-${count.index}
    spec:
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - key: app
                operator: In
                values:
                - ${local.bgsplit_trainer_tensorboard_app_name}
            topologyKey: kubernetes.io/hostname
      containers:
      - image: ${var.bgsplit_trainer_tensorboard_image_name}
        name: ${local.bgsplit_trainer_tensorboard_app_name}
        command: ["/bin/sh", "-c"]
        args: ["tensorboard --logdir ${local.nfs_mount_dir}/bgsplit_logs --bind_all"]
        resources:
          limits:
            cpu: "1.5"
            memory: "5G"
          requests:
            cpu: "1.5"
            memory: "5G"
        env:
        - name: PORT
          value: "${local.bgsplit_trainer_tensorboard_internal_port}"
        - name: PYTHONUNBUFFERED
          value: "1"
        - name: PYTHONIOENCODING
          value: "UTF-8"
        ports:
        - containerPort: ${local.bgsplit_trainer_tensorboard_internal_port}
        volumeMounts:
        - mountPath: ${local.nfs_mount_dir}
          name: ${local.nfs_volume_name}
      nodeSelector:
        cloud.google.com/gke-nodepool: ${var.bgsplit_trainer_tensorboard_node_pool_name}
      volumes:
      - name: ${local.nfs_volume_name}
        persistentVolumeClaim:
          claimName: ${kubernetes_persistent_volume_claim.nfs_claim.metadata.0.name}
      hostIPC: true

YAML

  depends_on = [google_container_cluster.cluster, google_container_node_pool.bgsplit_trainer_tensorboard_np]
}

resource "kubernetes_service" "bgsplit_trainer_tensorboard_svc" {
  count = var.bgsplit_trainer_tensorboard_num_nodes

  metadata {
    name = "bgsplit-trainer-tensorboard-dep-${count.index}-svc"
  }
  spec {
    selector = {
      app = "${local.bgsplit_trainer_tensorboard_app_name}-${count.index}"
    }
    port {
      port        = local.bgsplit_trainer_tensorboard_external_port
      target_port = local.bgsplit_trainer_tensorboard_internal_port
    }

    type = "LoadBalancer"
  }
}

output "bgsplit_trainer_tensorboard_urls" {
  value = [
    for svc in kubernetes_service.bgsplit_trainer_tensorboard_svc:
    "http://${svc.status.0.load_balancer.0.ingress.0.ip}:${local.bgsplit_trainer_tensorboard_external_port}"
  ]
}
