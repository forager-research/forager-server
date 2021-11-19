import React, { useState, useEffect, useReducer, useRef, useContext } from "react";
import useInterval from "react-useinterval";
import { InputGroup, InputGroupText, InputGroupAddon, Input } from "reactstrap";
import {
  Navbar,
  Button,
  Container,
  Popover,
  PopoverHeader,
  PopoverBody,
  Table,
  Form,
  FormGroup,
  Label,
  Progress
} from "reactstrap";
import { ConfirmModal, SignInModal } from "./components";
import { UserContext } from "./UserContext"
import BounceLoader from "react-spinners/BounceLoader";
import { Link } from "react-router-dom";
import { faPlus } from "@fortawesome/free-solid-svg-icons";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";

import fromPairs from "lodash/fromPairs";
import toPairs from "lodash/toPairs";

const JOB_STATUS_INTERVAL = 5000;

const DatasetList = () => {
  const unfinishedJobsReducer = (state, action) => {
    switch (action.type) {
      case "ADD_JOB": {
        let newState = { ...state };
        if (!(action.datasetName in newState)) {
          newState[action.datasetName] = {};
        }
        newState[action.datasetName][action.jobId] = action.status;
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

  const controllerRef = useRef();

  async function getJobStatus() {
    if (controllerRef.current) {
      controllerRef.current.abort();
    }
    const controller = new AbortController();
    controllerRef.current = controller;

    try {
      for (const datasetName in unfinishedJobs) {
        if (Object.keys(unfinishedJobs[datasetName]).length === 0) continue;

        const url = new URL(endpoints.inferenceStatus);
        let params = {
          job_ids: Object.keys(unfinishedJobs[datasetName]),
        };
        url.search = new URLSearchParams(params).toString();
        let statuses = await fetch(url, {
          method: "GET",
          signal: controllerRef.current?.signal
        }).then((r) => r.json());
        for (const jobId in statuses) {
          if (statuses[jobId]["finished"]) {
            console.log("remove job");
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
              status: statuses[jobId]
            });
          }
        }
      }
      controllerRef.current = null;
    } catch (e) {
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
            status: null
          });
        }
        if (createDatasetComputeClip) {
          const jobId = await computeEmbeddings(datasetName, "clip", "clip");
          unfinishedJobsDispatch({
            type: "ADD_JOB",
            datasetName,
            jobId,
            status: null
          });
        }

        getDatasetList();
        setCreateDatasetPopoverOpen(false);
      }
      setCreateDatasetLoading(false);
    };
    fn();
  };

  const [confirmIsOpen, setConfirmIsOpen] = useState(false);
  const [confirmDataset, setConfirmDataset] = useState(null);
  const toggleConfirmIsOpen = (category) => setConfirmIsOpen(!confirmIsOpen);

  const deleteDataset = (datasetName) => {
    const fn = async () => {
      const url = new URL(endpoints.deleteDataset);
      const body = {
        dataset: datasetName,
      };
      let resp = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }).then((r) => r.json());
      if (resp.status !== "success") {
        console.log(resp);
      }
      getDatasetList();
    };
    fn();
  };


  const { username, setUsername } = useContext(UserContext);
  const [isOpenSignIn, setIsOpenSignIn] = useState(false);
  const toggleSignIn = () => setIsOpenSignIn(!isOpenSignIn);

  useEffect(() => {
    if (username === null || username === undefined || username.length === 0) {
      setIsOpenSignIn(true);
    }
  }, []);

  return (
      <div>

      <Navbar>
        <Container fluid>
      <span>
      <h2>Forager</h2>
      </span>
      <span>
              {username ? (
                <>
                  {username} (
                  <a
                    href="#"
                    onClick={(e) => {
                      setUsername("");
                      setIsOpenSignIn(true);
                      e.preventDefault();
                    }}
                  >
                    Sign out
                  </a>
                  )
                </>
              ) : (
                <a
                  href="#"
                  onClick={(e) => {
                    setIsOpenSignIn(true);
                    e.preventDefault();
                  }}
                >
                  Sign in
                </a>
              )}
      </span>
        </Container>
    </Navbar>
    <Container>
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
                  d.name in unfinishedJobs && Object.keys(unfinishedJobs[d.name]).length > 0;
                let timeLeft = 0;
                let prog = 200;
                if (notReady) {
                  for (const key of Object.keys(unfinishedJobs[d.name])) {
                    if (unfinishedJobs[d.name][key] == null) continue;

                    if ("progress" in unfinishedJobs[d.name][key]["status"]) {
                      prog = Math.min(prog, (unfinishedJobs[d.name][key]["status"]["progress"]["n_processed"] /
                        unfinishedJobs[d.name][key]["status"]["progress"]["n_total"])
                        * 100);
                    }
                    timeLeft = Math.max(timeLeft, unfinishedJobs[d.name][key]["status"]["time_left"]);
                  }
                }
                if (prog === 200) {
                  prog = 0;
                }
                if (timeLeft === 0) {
                  timeLeft = "calculating...";
                } else {
                  timeLeft = new Date(timeLeft * 1000).toISOString().substr(11, 8);
                }
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
                    <td>{notReady ?
                         <>
                           <div>
                             <div className="text-center">
                               Time left: {timeLeft}
                             </div>
                             <Progress value={prog}/>
                           </div>
                         </>: "Ready"}</td>
                    <td>
                      <Button
                        close
                        onClick={(e) => {
                          setConfirmDataset(d.name);
                          toggleConfirmIsOpen();
                          document.activeElement.blur();
                        }}
                      />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </Table>
        ) : (
          <p>No datasets available.</p>
        )}
        <ConfirmModal
          isOpen={confirmIsOpen}
          toggle={toggleConfirmIsOpen}
          message={
            <span>
              Are you sure you want to delete the dataset{" "}
              <strong>{confirmDataset}</strong>? This action cannot be undone.
            </span>
          }
          confirmBtn={
            <Button
              color="danger"
              onClick={(e) => {
                deleteDataset(confirmDataset);
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
      <SignInModal isOpen={isOpenSignIn} toggle={toggleSignIn}/>
    </Container>
      </div>
  );
};

export default DatasetList;
