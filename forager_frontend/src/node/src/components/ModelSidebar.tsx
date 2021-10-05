import React, { useState, useEffect, useRef } from "react";
import useInterval from "react-useinterval";
import {
  Button,
  Input,
  FormGroup,
  Spinner,
  Collapse,
  Label,
  Form,
  CustomInput,
} from "reactstrap";
import { ReactSVG } from "react-svg";

import fromPairs from "lodash/fromPairs";
import toPairs from "lodash/toPairs";

import CategoryInput from "./CategoryInput";
import FeatureInput from "./FeatureInput";

import { DatasetInfo, ClusterInfo } from "../types";
import ClusterConnectionModal, {
  ClusterConnectionModalProps,
} from "./ClusterConnectionModal";

var dateFormat = require("dateformat");

const CLUSTER_STATUS_POLL_INTERVAL = 10000; // ms

const dnns = [{ id: "dnn", label: "DNN w/ BG Splitting" }];

const endpoints = fromPairs(
  toPairs({
    getModels: "get_models",
    trainModel: "train_model",
    modelStatus: "model",
    modelInference: "model_inference",
    modelInferenceStatus: "model_inference_status",
    stopModelInference: "stop_model_inference",
    startCluster: "start_cluster",
    clusterStatus: "cluster",
  }).map(([name, endpoint]) => [
    name,
    `${process.env.REACT_APP_SERVER_URL}/api/${endpoint}`,
  ])
);

export type ModelSidebarProps = {
  datasetName: string;
  datasetInfo: DatasetInfo;
  modelInfo: Any;
  setModelInfo: Any;
  isOpen: boolean;
  isVisible: boolean;
  username: string;
  disabled: boolean;
};

const ModelSidebar = ({
  datasetName,
  datasetInfo,
  modelInfo,
  setModelInfo,
  isOpen,
  isVisible,
  username,
  disabled,
}: ModelSidebarProps) => {
  // -------------------------------------------------------------------
  // CLUSTER
  // -------------------------------------------------------------------
  const [clusterInfo, setClusterInfo] = useState<ClusterInfo | null>(null);

  const [clusterConnectModalOpen, setClusterConnectModalOpen] =
    useState<boolean>(false);
  const toggleClusterConnectModal = () => {
    setClusterConnectModalOpen((open) => !open);
  };

  const [clusterId, setClusterId] = useState(null);

  const getClusterInfo = async () => {
    const url = new URL(`${endpoints.clusterStatus}/${clusterId}`);
    const clusterInfo = await fetch(url, {
      method: "GET",
    }).then((r) => r.json());
    setClusterInfo(clusterInfo);
  };

  const resetCluster = async () => {
    setClusterId(null);
    setClusterReady(false);
  };

  useEffect(() => {}, [clusterInfo]);
  useInterval(getClusterInfo, clusterId ? CLUSTER_STATUS_POLL_INTERVAL : null);

  if (!isVisible) return null;
  return (
    <Collapse className="model-sidebar" isOpen={isOpen}>
      <div>
        <div>
          {clusterInfo == null ||
          clusterInfo.status === ClusterConnectionStatus.DISCONNECTED ? (
            <Button color="light" onClick={toggleClusterConnectModal}>
              Connect...
            </Button>
          ) : (
            <div>Cluster status: {clusterInfo.status}</div>
          )}
        </div>
      </div>
      <ClusterConnectionModal
        isOpen={clusterConnectModalOpen}
        toggle={toggleClusterConnectModal}
        username={username}
        isReadOnly={!!!username}
        selectedClusterId={clusterId}
        setSelectedClusterId={setClusterId}
      />
    </Collapse>
  );
};

export default ModelSidebar;
