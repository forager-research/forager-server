import React, {
  useState,
  useCallback,
  useEffect,
  useMemo,
  useRef,
} from "react";
import useInterval from "react-useinterval";
import {
  Button,
  Form,
  FormGroup,
  Input,
  Table,
  Modal,
  ModalHeader,
  ModalFooter,
  ModalBody,
} from "reactstrap";

import { ClusterInfo } from "../types";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {
  faSort,
  faSortDown,
  faSortUp,
} from "@fortawesome/free-solid-svg-icons";

import styled from "styled-components";

import fromPairs from "lodash/fromPairs";
import sortBy from "lodash/sortBy";
import toPairs from "lodash/toPairs";

import { ConfirmModal } from "../components";

const endpoints = fromPairs(
  toPairs({
    startCluster: "start_cluster",
    stopCluster: "stop_cluster",
    clusterStatus: "cluster",
  }).map(([name, endpoint]) => [
    name,
    `${process.env.REACT_APP_SERVER_URL}/api/${endpoint}`,
  ])
);

const TableContainer = styled.div`
  height: 70vh;
  overflow: auto;
`;

const CLUSTER_STATUS_POLL_INTERVAL = 3000; // ms

export type ClusterConnectionModalProps = {
  isOpen: boolean;
  toggle: () => void;
  username: string;
  isReadOnly: boolean;
  selectedClusterId: string;
  setSelectedClusterId: (clusterId: string) => void;
};

const ClusterConnectionModal = (props: ClusterConnectionModalProps) => {
  const [clusterStatuses, setClusterStatuses] = useState<ClusterInfo[]>([]);
  const getClusterStatuses = () => {
    const fn = async () => {
      const url = new URL(`${endpoints.clusterStatus}`);
      const clusterStatuses = await fetch(url, {
        method: "GET",
      }).then((r) => r.json());
      setClusterStatuses(clusterStatuses);
    };
    fn();
  };
  useInterval(getClusterStatuses, CLUSTER_STATUS_POLL_INTERVAL);

  const [hideOthersTags, setHideOthersTags] = useState<boolean>(false);

  const [confirmIsOpen, setConfirmIsOpen] = useState<boolean>(false);
  const [confirmCluster, setConfirmCluster] = useState<string>("");
  const [confirmClusterIdx, setConfirmClusterIdx] = useState<number>(-1);
  const toggleConfirmIsOpen = () => setConfirmIsOpen(!confirmIsOpen);

  const createCluster = async () => {
    if (props.isReadOnly) return;

    const url = new URL(endpoints.startCluster);
    const body = {};

    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((res) => res.json());
    props.setSelectedClusterId(res["cluster_id"]);
    props.toggle();
  };

  const stopClusterByIndex = async (idx) => {
    if (props.isReadOnly) return;

    const url = new URL(endpoints.stopCluster);
    const body = {
      cluster_id: clusterStatuses[idx].id,
    };

    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((res) => res.json());
  };

  const tableBodyFromTags = () => {
    return (
      <tbody>
        {clusterStatuses.map((info: ClusterInfo, i) => {
          return (
            <tr key={info.id}>
              <td>
                <Button
                  disabled={props.isReadOnly}
                  onClick={(e) => {
                    props.setSelectedClusterId(info.id);
                    props.toggle();
                  }}
                >
                  Select
                </Button>
              </td>
              <td>
                <input type="text" value={info.name} disabled={true} />
              </td>
              <td>{info.id}</td>
              <td>{info.created_at}</td>
              <td>{info.status}</td>
              <td>
                <Button
                  close
                  disabled={props.isReadOnly}
                  onClick={(e) => {
                    setConfirmCluster(obj.name);
                    setConfirmClusterIdx(i);
                    toggleConfirmIsOpen();
                    document.activeElement.blur();
                  }}
                />
              </td>
            </tr>
          );
        })}
      </tbody>
    );
  };

  return (
    <Modal
      isOpen={props.isOpen}
      toggle={props.toggle}
      modalTransition={{ timeout: 25 }}
      backdropTransition={{ timeout: 75 }}
      size="lg"
    >
      <ModalHeader toggle={props.toggle}>Manage Clusters</ModalHeader>
      <ModalBody>
        <div className="custom-switch custom-control mr-4">
          <Input
            type="checkbox"
            className="custom-control-input"
            id="hide-others-tags-switch"
            checked={hideOthersTags}
            onChange={(e) => setHideOthersTags(e.target.checked)}
          />
          <label
            className="custom-control-label text-nowrap"
            htmlFor="hide-others-tags-switch"
          >
            Hide clusters from others
          </label>
        </div>
        <TableContainer className="pr-2">
          <Table hover borderless size="sm">
            <thead>
              <tr>
                <th>Name</th>
                <th>ID</th>
                <th>Created At</th>
                <th>Status </th>
              </tr>
            </thead>
            {tableBodyFromTags()}
          </Table>
        </TableContainer>

        <Button
          color="simple"
          onClick={(e) => {
            createCluster();
          }}
        >
          New cluster
        </Button>

        <ConfirmModal
          isOpen={confirmIsOpen}
          toggle={toggleConfirmIsOpen}
          message={
            <span>
              Are you sure you want to delete the cluster{" "}
              <strong>{confirmCluster}</strong>?
            </span>
          }
          confirmBtn={
            <Button
              color="danger"
              onClick={(e) => {
                stopClusterByIndex(confirmClusterIdx);
                toggleConfirmIsOpen();
              }}
            >
              Delete
            </Button>
          }
          cancelBtn={
            <Button color="light" onClick={(e) => toggleConfirmIsOpen()}>
              Cancel
            </Button>
          }
        />
      </ModalBody>
    </Modal>
  );
};

export default ClusterConnectionModal;
