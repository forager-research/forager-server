import React, { useState, useCallback, useEffect, useMemo, useRef } from "react";
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

import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {
  faSort,
  faSortDown,
  faSortUp
} from "@fortawesome/free-solid-svg-icons";

import styled from "styled-components";

import ReactTimeAgo from "react-time-ago";
import fromPairs from "lodash/fromPairs";
import toPairs from "lodash/toPairs";

import { ConfirmModal } from "../components";

const endpoints = fromPairs(toPairs({
  deleteModelOutput: 'delete_model_output',
  getModelOutputs: "get_model_outputs",
}).map(([name, endpoint]) => [name, `${process.env.REACT_APP_SERVER_URL}/api/${endpoint}`]));

const TableContainer = styled.div`
  height: 70vh;
  overflow: auto;
`;

const ModelOutputManagementModal = ({
  isOpen,
  toggle,
  datasetName,
  modelOutputInfo,
  setModelOutputInfo,
  username,
  isReadOnly
}) => {

  // TODO: store with redux
  const [modelOutputs, setModelOutputs] = useState([]);

  const kOrderBy = fromPairs([["name", 0], ["timestamp", 1]]);
  const [orderBy, setOrderBy] = useState(kOrderBy.name);
  const [orderAscending, setOrderAscending] = useState(true);

  const [confirmIsOpen, setConfirmIsOpen] = useState(false);
  const [confirmModel, setConfirmModel] = useState(null);
  const [confirmModelIdx, setConfirmModelIdx] = useState(null);
  const toggleConfirmIsOpen = (model) => setConfirmIsOpen(!confirmIsOpen);

  const sortModelOutputs = (arr) => {
    const copy = arr.slice(0);
    const m = orderAscending ? 1 : -1;
    if (orderBy === kOrderBy.name) {
      copy.sort((a, b) => m * (a.modelOutput.name.toLowerCase() < b.modelOutput.name.toLowerCase() ? -1 : 1));
    } else if (orderBy === kOrderBy.timestamp) {
      copy.sort((a, b) => m * (a.modelOutput.timestamp < b.modelOutput.timestamp ? -1 : 1));
    }

    return copy;
  };

  const changeOrdering = (by) => {
    if (by === orderBy) {
      setOrderAscending(!orderAscending);
    } else {
      setOrderBy(by);
      setOrderAscending(true);
    }
  };

  const deleteModelOutputByIndex = async (idx) => {
    if (isReadOnly) return;

    const id = modelOutputs[idx].modelOutput.id;
    const url = new URL(endpoints.deleteModelOutput);
    const body = {
      user: username,
      model_output_id: id,
    };

    const res = await fetch(url, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    }).then(res => res.json());

    let newModelOutputInfo = [...modelOutputInfo];
    delete newModelOutputInfo[modelOutputs[idx].srcIdx];
    setModelOutputInfo(newModelOutputInfo);
  };

  const getModelOutputs = async () => {
    const url = new URL(`${endpoints.getModelOutputs}/${datasetName}`);
    const res = await fetch(url, {
      method: "GET",
      headers: {"Content-Type": "application/json"},
    }).then(res => res.json());
    console.log("got model outputs", res);
    setModelOutputInfo(res.model_outputs);
  }

  useEffect(() => {
    getModelOutputs();
  }, []);

  useEffect(() => {
    const fn = async () => {
      setModelOutputs(sortModelOutputs(modelOutputInfo.map(
        (modelOutput, i) => ({modelOutput, srcIdx: i})
      ).map((obj, i) => {
        if (!(obj.modelOutput.timestamp instanceof Date)) {
          obj.modelOutput.timestamp = new Date(obj.modelOutput.timestamp);
        }
        return obj;
      })))
    };
    fn();
  }, [modelOutputInfo]);

  useEffect(() => {
    setModelOutputs(prev => sortModelOutputs(prev))
  }, [orderBy, orderAscending]);

  const tableBodyFromModelOutputs = () => {
    return (
      <tbody>
        {modelOutputs.map((obj, i) => {
          return (
            <tr key={obj.srcIdx}>
              <td>
                <input
                  type="text"
                  value={obj.modelOutput.name}
                  disabled
                  onKeyDown={(e) => { if (e.keyCode === 13) e.target.blur(); }}
                />
              </td>
              <td>
                <ReactTimeAgo date={obj.modelOutput.timestamp} timeStyle="mini"/> ago
              </td>
              <td>
                <Button close disabled={isReadOnly} onClick={(e) => {
                    setConfirmModel(obj.modelOutput.name);
                    setConfirmModelIdx(i);
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
  }

  return (
    <Modal
      isOpen={isOpen}
      toggle={toggle}
      modalTransition={{ timeout: 25 }}
      backdropTransition={{ timeout: 75 }}
      size="lg"
    >
      <ModalHeader toggle={toggle}>
        Manage Model Outputs
      </ModalHeader>
      <ModalBody>
        <TableContainer className="pr-2">
          <Table hover borderless size="sm">
            <thead>
              <tr>
                <th style={{cursor: "pointer"}} onClick={() => changeOrdering(kOrderBy.name)}>
                  Model Output Name <FontAwesomeIcon icon={
                    orderBy !== kOrderBy.name ? faSort : (orderAscending ? faSortUp : faSortDown)
                  } />
                </th>
                <th style={{cursor: "pointer"}} onClick={() => changeOrdering(kOrderBy.timestamp)}>
                  Timestamp <FontAwesomeIcon icon={
                    orderBy !== kOrderBy.timestamp ? faSort : (orderAscending ? faSortUp : faSortDown)
                  } />
                </th>
              </tr>
            </thead>
            {tableBodyFromModelOutputs()}
          </Table>
        </TableContainer>

        <ConfirmModal
          isOpen={confirmIsOpen}
          toggle={toggleConfirmIsOpen}
          message={<span>Are you sure you want to delete the model output <strong>{confirmModel}</strong>? This action cannot be undone.</span>}
          confirmBtn={<Button color="danger" onClick={(e) => {
            deleteModelOutputByIndex(confirmModelIdx);
            toggleConfirmIsOpen();
          }}>Delete</Button>}
          cancelBtn={<Button color="light" onClick={(e) => toggleConfirmIsOpen()}>Cancel</Button>}
        />
      </ModalBody>
    </Modal>
  );
};

export default ModelOutputManagementModal;
