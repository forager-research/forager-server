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
  updateModel: 'update_model',
  deleteModel: 'delete_model',
  getModels: "get_models",
}).map(([name, endpoint]) => [name, `${process.env.REACT_APP_SERVER_URL}/api/${endpoint}`]));

const TableContainer = styled.div`
  height: 70vh;
  overflow: auto;
`;

const ModelManagementModal = ({
  isOpen,
  toggle,
  datasetName,
  modelInfo,
  setModelInfo,
  username,
  isReadOnly
}) => {

  // TODO: store with redux
  const [models, setModels] = useState([]);

  const kOrderBy = fromPairs([["name", 0], ["timestamp", 1]]);
  const [orderBy, setOrderBy] = useState(kOrderBy.name);
  const [orderAscending, setOrderAscending] = useState(true);

  const [confirmIsOpen, setConfirmIsOpen] = useState(false);
  const [confirmModel, setConfirmModel] = useState(null);
  const [confirmModelIdx, setConfirmModelIdx] = useState(null);
  const toggleConfirmIsOpen = (model) => setConfirmIsOpen(!confirmIsOpen);

  const sortModels = (arr) => {
    const copy = arr.slice(0);
    const m = orderAscending ? 1 : -1;
    if (orderBy === kOrderBy.name) {
      copy.sort((a, b) => m * (a.model.name.toLowerCase() < b.model.name.toLowerCase() ? -1 : 1));
    } else if (orderBy === kOrderBy.timestamp) {
      copy.sort((a, b) => m * (a.model.latest.timestamp < b.model.latest.timestamp ? -1 : 1));
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

  const setModelByIndex = (name, idx) => {
    if (isReadOnly) return;

    models[idx].name = name;
    setModels(models.slice(0));
  };

  const updateModelByIndex = async (e, idx, srcIdx) => {
    if (isReadOnly) return;

    const newName = models[idx].name;
    const oldName = modelInfo[srcIdx];
    if (newName === oldName) return;

    if (modelInfo.find((model) => model.name === newName) !== undefined) {
      e.target.setCustomValidity(`"${newName}" already exists`);
      e.target.reportValidity();
      models[idx].name = oldName;
      setModels(models.slice(0));
      return;
    }

    const url = new URL(endpoints.updateModel);
    const body = {
      user: username,
      new_model_name: newName,
      old_model_name: oldName,
    };

    const res = await fetch(url, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    }).then(res => res.json());

    const newModelInfo = [...modelInfo];
    newModelInfo[srcIdx].name = newName
    setModelInfo(newModelInfo);
  };

  const deleteModelByIndex = async (idx) => {
    if (isReadOnly) return;

    const name = models[idx].model.name;
    const url = new URL(endpoints.deleteModel);
    const body = {
      user: username,
      model_name: name,
    };

    const res = await fetch(url, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    }).then(res => res.json());

    let newModelInfo = [...modelInfo];
    delete newModelInfo[models[idx].srcIdx];
    setModelInfo(newModelInfo);
  };

  useEffect(() => {
    const fn = async () => {
      setModels(sortModels(modelInfo.map(
        (model, i) => ({model, srcIdx: i})
      ).map((obj, i) => {
        if (!(obj.model.latest.timestamp instanceof Date)) {
          obj.model.latest.timestamp = new Date(obj.model.latest.timestamp);
        }
        return obj;
      })));
    };
    fn();
  }, [modelInfo]);

  useEffect(() => {
    setModels(prev => sortModels(prev))
  }, [orderBy, orderAscending]);

  const tableBodyFromModels = () => {
    return (
      <tbody>
        {models.map((obj, i) => {
          return (
            <tr key={obj.srcIdx}>
              <td>
                <input
                  type="text"
                  value={obj.model.name}
                  disabled={isReadOnly}
                  onChange={(e) => setModelByIndex(e.target.value, i)}
                  onKeyDown={(e) => { if (e.keyCode === 13) e.target.blur(); }}
                  onBlur={(e) => updateModelByIndex(e, i, obj.srcIdx)}
                />
              </td>
              <td>{obj.model.latest.epoch}</td>
              <td>
                <ReactTimeAgo date={obj.model.latest.timestamp} timeStyle="mini"/> ago
              </td>
              <td>
                {obj.model.with_output != null ? 'Yes' : 'No'}
              </td>
              <td>
                <Button close disabled={isReadOnly} onClick={(e) => {
                    setConfirmModel(obj.model.name);
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
        Manage Models
      </ModalHeader>
      <ModalBody>
        <TableContainer className="pr-2">
          <Table hover borderless size="sm">
            <thead>
              <tr>
                <th style={{cursor: "pointer"}} onClick={() => changeOrdering(kOrderBy.name)}>
                  Model Name <FontAwesomeIcon icon={
                    orderBy !== kOrderBy.name ? faSort : (orderAscending ? faSortUp : faSortDown)
                  } />
                </th>
                <th>
                  Epoch
                </th>
                <th style={{cursor: "pointer"}} onClick={() => changeOrdering(kOrderBy.timestamp)}>
                  Timestamp <FontAwesomeIcon icon={
                    orderBy !== kOrderBy.timestamp ? faSort : (orderAscending ? faSortUp : faSortDown)
                  } />
                </th>
                <th>
                  Has Output?
                </th>
              </tr>
            </thead>
            {tableBodyFromModels()}
          </Table>
        </TableContainer>

        <ConfirmModal
          isOpen={confirmIsOpen}
          toggle={toggleConfirmIsOpen}
          message={<span>Are you sure you want to delete the model <strong>{confirmModel}</strong>? This action cannot be undone.</span>}
          confirmBtn={<Button color="danger" onClick={(e) => {
            deleteModelByIndex(confirmModelIdx);
            toggleConfirmIsOpen();
          }}>Delete</Button>}
          cancelBtn={<Button color="light" onClick={(e) => toggleConfirmIsOpen()}>Cancel</Button>}
        />
      </ModalBody>
    </Modal>
  );
};

export default ModelManagementModal;
