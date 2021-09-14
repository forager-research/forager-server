variable "project_id" {
  type = string
}

variable "zone" {
  type = string
}

variable "network" {
  type = string
}

provider "google" {
  project = var.project_id
}

resource "random_uuid" "cluster_id" { }
