import React, { useState, useEffect, useReducer } from "react";
import useInterval from "react-useinterval";
import { InputGroup, InputGroupText, InputGroupAddon, Input } from "reactstrap";
import {
  Button,
  Container,
  Popover,
  PopoverHeader,
  PopoverBody,
  Table,
  Form,
  FormGroup,
  Label,
} from "reactstrap";
import BounceLoader from "react-spinners/BounceLoader";
import { Link } from "react-router-dom";
import { faPlus } from "@fortawesome/free-solid-svg-icons";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";

import fromPairs from "lodash/fromPairs";
import toPairs from "lodash/toPairs";

const JOB_STATUS_INTERVAL = 5000;

const endpoints = fromPairs(
  toPairs({
    getDatasets: "get_datasets",
    createDataset: "create_dataset",
    startInference: "start_model_inference",
    inferenceStatus: "model_inference_status",
  }).map(([name, endpoint]) => [
    name,
    `${process.env.REACT_APP_SERVER_URL}/api/${endpoint}`,
  ])
);

const DatasetList = () => {
  const unfinishedJobsReducer = (state, action) => {
    switch (action.type) {
      case "ADD_JOB": {
        let newState = { ...state };
        if (!(action.datasetName in newState)) {
          newState[action.datasetName] = {};
        }
        newState[action.datasetName][action.jobId] = true;
        return newState;
      }
      case "REMOVE_JOB": {
        let newState = { ...state };
        if (
          action.datasetName in newState &&
          action.jobId in newState[action.datasetName]
        ) {
          delete newState[action.datasetName][action.jobId];
        }
        return newState;
      }
      default:
        throw new Error();
    }
  };

  const [datasets, setDatasets] = useState([]);
  const [unfinishedJobs, unfinishedJobsDispatch] = useReducer(
    unfinishedJobsReducer,
    {}
  );

  async function getDatasetList() {
    const url = new URL(endpoints.getDatasets);
    let _datasets = await fetch(url, {
      method: "GET",
    }).then((r) => r.json());
    setDatasets(_datasets.datasets);
  }

  useEffect(() => {
    getDatasetList();
  }, []);

  async function getJobStatus() {
    for (const datasetName in unfinishedJobs) {
      if (Object.keys(unfinishedJobs[datasetName]).length === 0) continue;

      const url = new URL(endpoints.inferenceStatus);
      let params = {
        job_ids: Object.keys(unfinishedJobs[datasetName]),
      };
      url.search = new URLSearchParams(params).toString();
      let statuses = await fetch(url, {
        method: "GET",
      }).then((r) => r.json());
      for (const jobId in statuses) {
        if (statuses[jobId]["finished"]) {
          unfinishedJobsDispatch({
            type: "REMOVE_JOB",
            datasetName,
            jobId,
          });
        } else {
          unfinishedJobsDispatch({
            type: "ADD_JOB",
            datasetName,
            jobId,
          });
        }
      }
    }
  }

  useInterval(() => {
    getJobStatus();
  }, JOB_STATUS_INTERVAL);

  const [createDatasetPopoverOpen, setCreateDatasetPopoverOpen] =
    useState(false);
  const [createDatasetName, setCreateDatasetName] = useState("");
  const [createDatasetTrainDirectory, setCreateDatasetTrainDirectory] =
    useState("");
  const [createDatasetValDirectory, setCreateDatasetValDirectory] =
    useState("");
  const [createDatasetComputeResnet, setCreateDatasetComputeResnet] =
    useState(true);
  const [createDatasetComputeClip, setCreateDatasetComputeClip] =
    useState(true);
  const [createDatasetLoading, setCreateDatasetLoading] = useState(false);
  const toggleCreateDatasetPopoverOpen = () => {
    setCreateDatasetPopoverOpen((value) => !value);
  };

  const computeEmbeddings = async (datasetName, modelType, modelOutputName) => {
    const url = new URL(endpoints.startInference);
    const body = {
      dataset: datasetName,
      model: modelType,
      model_output_name: modelOutputName,
    };
    let resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => r.json());
    return resp["job_id"];
  };

  const createDataset = (e) => {
    e.preventDefault();
    const fn = async () => {
      const url = new URL(endpoints.createDataset);
      const datasetName = e.target.datasetName.value;
      const body = {
        dataset: datasetName,
        train_images_directory: e.target.trainDirectoryPath.value,
        val_images_directory: e.target.valDirectoryPath.value,
      };
      setCreateDatasetLoading(true);
      let resp = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }).then((r) => r.json());
      if (resp.status !== "success") {
        console.log(resp);
      } else {
        const jobs = { ...unfinishedJobs };
        jobs[datasetName] = [];
        if (createDatasetComputeResnet) {
          const jobId = await computeEmbeddings(
            datasetName,
            "resnet",
            "resnet"
          );
          unfinishedJobsDispatch({
            type: "ADD_JOB",
            datasetName,
            jobId,
          });
        }
        if (createDatasetComputeClip) {
          const jobId = await computeEmbeddings(datasetName, "clip", "clip");
          unfinishedJobsDispatch({
            type: "ADD_JOB",
            datasetName,
            jobId,
          });
        }

        getDatasetList();
        setCreateDatasetPopoverOpen(false);
      }
      setCreateDatasetLoading(false);
    };
    fn();
  };

  return (
    <Container>
      <h2>Forager</h2>
      <div>
        <h3>Datasets</h3>
        {datasets.length > 0 ? (
          <Table borderless size="sm">
            <thead>
              <tr>
                <th>Name</th>
                <th>Created</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {datasets.map((d) => {
                const notReady =
                  d.name in unfinishedJobs && unfinishedJobs[d.name].length > 0;
                return (
                  <tr key={d.name}>
                    <td>
                      {notReady ? (
                        d.name
                      ) : (
                        <Link to={`/${d.name}`} activeClassName="active">
                          {d.name}
                        </Link>
                      )}
                    </td>
                    <td>{d.created_at}</td>
                    <td>{notReady ? "Setting up..." : "Ready"}</td>
                  </tr>
                );
              })}
            </tbody>
          </Table>
        ) : (
          <p>No datasets available.</p>
        )}
        <div className="create-dataset-container">
          <Button
            id="create-dataset-open-button"
            color="light"
            size="md"
            type="button"
          >
            <FontAwesomeIcon icon={faPlus} className="mr-1" />
            Add dataset
          </Button>
          <Popover
            placement="right-start"
            target="create-dataset-open-button"
            isOpen={createDatasetPopoverOpen}
            toggle={toggleCreateDatasetPopoverOpen}
            trigger="legacy"
          >
            <PopoverHeader>Create Dataset</PopoverHeader>
            <PopoverBody>
              <Form onSubmit={createDataset}>
                <FormGroup>
                  <Label for="datasetNameInput">Dataset Name</Label>
                  <Input type="text" name="datasetName" id="datasetNameInput" />
                </FormGroup>
                <FormGroup>
                  <Label for="trainDirectory">Train Images Directory</Label>
                  <Input
                    type="text"
                    name="trainDirectoryPath"
                    id="trainDirectory"
                  />
                </FormGroup>
                <FormGroup>
                  <Label for="valDirectory">Validation Images Directory</Label>
                  <Input
                    type="text"
                    name="valDirectoryPath"
                    id="valDirectory"
                  />
                </FormGroup>
                <div className="custom-switch custom-control mb-1">
                  <Input
                    type="checkbox"
                    className="custom-control-input"
                    id="compute-resnet-embeddings-switch"
                    checked={createDatasetComputeResnet}
                    onChange={(e) =>
                      setCreateDatasetComputeResnet(e.target.checked)
                    }
                  />
                  <label
                    className="custom-control-label text-nowrap"
                    htmlFor="compute-resnet-embeddings-switch"
                  >
                    ResNet embeddings
                  </label>
                </div>
                <div className="custom-switch custom-control mb-1">
                  <Input
                    type="checkbox"
                    className="custom-control-input"
                    id="compute-clip-embeddings-switch"
                    checked={createDatasetComputeClip}
                    onChange={(e) =>
                      setCreateDatasetComputeClip(e.target.checked)
                    }
                  />
                  <label
                    className="custom-control-label text-nowrap"
                    htmlFor="compute-clip-embeddings-switch"
                  >
                    Clip embeddings
                  </label>
                </div>
                <Button size="sm" type="submit" disabled={createDatasetLoading}>
                  Create{" "}
                </Button>
                <BounceLoader
                  css={{ position: "absolute", marginLeft: "2em" }}
                  size={30}
                  loading={createDatasetLoading}
                  color="purple"
                />
              </Form>
            </PopoverBody>
          </Popover>
        </div>
      </div>
    </Container>
  );
};

export default DatasetList;
