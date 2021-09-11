import React, { useState, useEffect, useRef } from "react";
import useInterval from 'react-useinterval';
import {
  Button,
  Input,
  FormGroup,
  Spinner,
  Collapse,
  Label,
  Form,
  CustomInput
} from "reactstrap";
import { ReactSVG } from "react-svg";

import fromPairs from "lodash/fromPairs";
import toPairs from "lodash/toPairs";

import CategoryInput from "./CategoryInput";
import FeatureInput from "./FeatureInput";

var dateFormat = require("dateformat");

const STATUS_POLL_INTERVAL = 3000;  // ms

const dnns = [
  {id: "dnn", label: "DNN w/ BG Splitting"},
];

const endpoints = fromPairs(toPairs({
  getModels: "get_models",
  trainModel: "train_model",
  modelStatus: "model",
  modelInference: "model_inference",
  modelInferenceStatus: "model_inference_status",
  stopModelInference: "stop_model_inference",
  startCluster: "start_cluster",
  clusterStatus: "cluster",
}).map(([name, endpoint]) => [name, `${process.env.REACT_APP_SERVER_URL}/api/${endpoint}`]));

const TrainPanel = ({
  datasetName,
  datasetInfo,
  modelInfo,
  setModelInfo,
  isVisible,
  username,
  disabled,
  customModesByCategory,
}) => {
  const [dnnAdvancedIsOpen, setDnnAdvancedIsOpen] = useState(false);

  const [modelName, setModelName] = useState();
  const [dnnType, setDnnType] = useState();
  const [dnnAugmentNegs, setDnnAugmentNegs] = useState();
  const [dnnPosTags, setDnnPosTags] = useState([]);
  const [dnnNegTags, setDnnNegTags] = useState([]);
  const [dnnValPosTags, setDnnValPosTags] = useState([]);
  const [dnnValNegTags, setDnnValNegTags] = useState([]);

  const [dnnCheckpointModel, setDnnCheckpointModel] = useState();
  const [dnnAugmentIncludeTags, setDnnAugmentIncludeTags] = useState([]);
  const [dnnAugmentExcludeTags, setDnnAugmentExcludeTags] = useState([]);

  //
  // CLUSTER
  //
  const [clusterId, setClusterId] = useState(null);
  const [clusterReady, setClusterReady] = useState(false);

  const startCreatingCluster = async () => {
    const url = new URL(`${endpoints.startCluster}`);
    var clusterResponse = await fetch(url, {
      method: "POST",
    }).then(r => r.json());
    setClusterId(clusterResponse.cluster_id);
    setClusterReady(false);
  };

  const getClusterStatus = async () => {
    const url = new URL(`${endpoints.clusterStatus}/${clusterId}`);
    const clusterStatus = await fetch(url, {
      method: "GET",
    }).then(r => r.json());
    setClusterReady(clusterStatus.ready);
  };

  const resetCluster = async () => {
    setClusterId(null);
    setClusterReady(false);
  };

  useInterval(
    getClusterStatus,
    clusterId && !clusterReady ? STATUS_POLL_INTERVAL : null);

  //
  // TRAINING
  //
  const [requestDnnTraining, setRequestDnnTraining] = useState(false);
  const [dnnIsTraining, setDnnIsTraining] = useState(false);
  const [trainingModelId, setTrainingModelId] = useState();
  const [prevModelId, setPrevModelId] = useState();
  const [trainingEpoch, setTrainingEpoch] = useState();
  const [trainingTimeLeft, setTrainingTimeLeft] = useState();
  const [trainingTensorboardUrl, setTrainingTensorboardUrl] = useState();
  const [dnnKwargs, setDnnKwargs] = useState({
    "initial_lr": 0.01,
    "endlr": 0.0,
    "max_epochs": 90,
    "warmup_epochs": 0,
    "epochs_to_run": 5,
    "input_size": 256,
    "batch_size": 256,
    "momentum": 0.9,
    "weight_decay": 0.0001,
    "aux_weight": 0.04,
    "aux_labels_type": "imagenet",
    "restrict_aux_labels": true,
    "freeze_backbone": false,
    "cache_images_on_disk": true,
  });
  let optInputs = [
    {type: "number", displayName: "Initial LR", param: "initial_lr"},
    {type: "number", displayName: "End LR", param: "endlr"},
    {type: "number", displayName: "Momentum", param: "momentum"},
    {type: "number", displayName: "Weight decay", param: "weight_decay"},
    {type: "number", displayName: "Warmup epochs", param: "warmup_epochs"},
    {type: "number", displayName: "Max epochs", param: "max_epochs"},
    {type: "number", displayName: "Epochs per job", param: "epochs_to_run"},
  ]
  let otherInputs = [
    {type: "number", displayName: "Image input size", param: "input_size"},
    {type: "number", displayName: "Batch size", param: "batch_size"},
    {type: "number", displayName: "Aux loss weight", param: "aux_weight"},
    {type: "checkbox", displayName: "Restrict aux labels", param: "restrict_aux_labels"},
    {type: "checkbox", displayName: "Freeze backbone", param: "freeze_backbone"},
    {type: "checkbox", displayName: "Cache images on disk", param: "cache_images_on_disk"},
  ]

  const updateDnnKwargs = (param, value) => {
    setDnnKwargs(state => {
      return {...state, [param]: value};
    });
  };

  const handleDnnKwargsChange = (e) => {
    var param = e.target.name;
    if (e.target.type === "checkbox" ||
        e.target.type === "switch") {
      var value = e.target.checked;
    } else {
      var value = parseFloat(e.target.value);
    }
    updateDnnKwargs(param, value);
  };

  const trainDnnOneEpoch = async () => {
    const url = new URL(`${endpoints.trainModel}/${datasetName}`);
    let body = {
      model_name: modelName,
      cluster_id: clusterId,
      bucket: "foragerml",
      index_id: datasetInfo.index_id,
      pos_tags: dnnPosTags.map(t => `${t.category}:${t.value}`),
      neg_tags: dnnNegTags.map(t => `${t.category}:${t.value}`),
      val_pos_tags: dnnValPosTags.map(t => `${t.category}:${t.value}`),
      val_neg_tags: dnnValNegTags.map(t => `${t.category}:${t.value}`),
      augment_negs: dnnAugmentNegs,
      include: dnnAugmentIncludeTags.map(t => `${t.category}:${t.value}`),
      exclude: dnnAugmentExcludeTags.map(t => `${t.category}:${t.value}`),
      model_kwargs: dnnKwargs,
    }
    if (prevModelId) {
      body.resume = prevModelId;
      body.model_kwargs.resume_training = true;
    } else if (dnnCheckpointModel) {
      body.resume = dnnCheckpointModel.latest.model_id;
      body.model_kwargs.resume_training = false;
    }

    const r = await fetch(url, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    });
    let data = await r.json();
    if (r.ok) {
      setTrainingModelId(data.model_id);
    } else {
      if (data.reason.startsWith("Cluster")) {
        await resetCluster();
      } else {
        console.error("Model training did not start", data.reason);
        setRequestDnnTraining(false);
      }
      setDnnIsTraining(false);
    }
  };

  const getTrainingStatus = async () => {
    if (!trainingModelId) return;
    const url = new URL(`${endpoints.modelStatus}/${trainingModelId}`);
    const modelStatus = await fetch(url, {
      method: "GET",
    }).then(r => r.json());
    if (modelStatus.has_model && dnnIsTraining) {
      console.log("training status next epoch",
                  new Date().toLocaleTimeString(),
                  dnnIsTraining, calledDnnTrain.current,
                  trainingModelId)
      // Start next epoch
      setPrevModelId(trainingModelId);
      setTrainingModelId(null);
      if (trainingEpoch + 1 >= dnnKwargs["max_epochs"]) {
        setRequestDnnTraining(false);
      } else {
        setTrainingEpoch(
          Math.min(dnnKwargs["max_epochs"] - 1,
                   trainingEpoch + dnnKwargs["epochs_to_run"]));
      }
      setDnnIsTraining(false);
    } else if (modelStatus.failed && dnnIsTraining) {
      console.error("Model training failed", modelStatus.failure_reason);
      setRequestDnnTraining(false);
      setDnnIsTraining(false);
      setTrainingModelId(null);
    }
    setTrainingTimeLeft(modelStatus.training_time_left);
    setTrainingTensorboardUrl(modelStatus.tensorboard_url);
  };

  useInterval(
    getTrainingStatus,
    dnnIsTraining && trainingModelId ? STATUS_POLL_INTERVAL : null);

  // Start training
  useEffect(() => {
    if (!requestDnnTraining) return;
    if (clusterReady && !dnnIsTraining) {
      setDnnIsTraining(true);
    } else if (!clusterId) {
      startCreatingCluster();
    }
  }, [clusterReady, clusterId, requestDnnTraining, dnnIsTraining]);

  const calledDnnTrain = useRef(false);
  useEffect(() => {
    console.log("dnn call",
                new Date().toLocaleTimeString(),
                dnnIsTraining, calledDnnTrain.current)
    if (dnnIsTraining && !calledDnnTrain.current) {
      calledDnnTrain.current = true;
      trainDnnOneEpoch();
    } else {
      calledDnnTrain.current = false;
    }
  }, [dnnIsTraining]);

  //
  // INFERENCE (automatically pipelined with training)
  //
  const [dnnIsInferring, setDnnIsInferring] = useState(false);
  const [inferenceJobId, setInferenceJobId] = useState(null);
  const [inferenceEpoch, setInferenceEpoch] = useState();
  const [inferenceTimeLeft, setInferenceTimeLeft] = useState();

  const inferOneEpoch = async () => {
    const url = new URL(`${endpoints.modelInference}/${datasetName}`);
    let body = {
      model_id: prevModelId,
      cluster_id: clusterId,
      bucket: "foragerml",
      index_id: datasetInfo.index_id,
    }
    const inferenceResponse = await fetch(url, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    }).then(r => r.json());

    setInferenceEpoch(trainingEpoch - 1);
    setInferenceJobId(inferenceResponse.job_id);
  };

  const stopInference = async () => {
    const url = new URL(`${endpoints.stopModelInference}/${inferenceJobId}`);
    await fetch(url, {
      method: "POST",
    }).then(r => r.json());
    setInferenceJobId(null);
    setDnnIsInferring(false);
  };

  const getInferenceStatus = async () => {
    const url = new URL(`${endpoints.modelInferenceStatus}/${inferenceJobId}`);
    const inferenceStatus = await fetch(url, {
      method: "GET",
    }).then(r => r.json());
    if (inferenceStatus.has_output) {
      setDnnIsInferring(false);
      setInferenceJobId(null);
    }
    setInferenceTimeLeft(inferenceStatus.time_left);
  };

  useInterval(
    getInferenceStatus,
    inferenceJobId ? STATUS_POLL_INTERVAL : null);

  useEffect(() => {
    if (prevModelId && !inferenceJobId && trainingEpoch > inferenceEpoch + 1 && !dnnIsInferring) {
      setDnnIsInferring(true);
    }
  }, [prevModelId, inferenceJobId, trainingEpoch, inferenceEpoch]);

  const calledDnnInfer = useRef(false);
  useEffect(() => {
    if (dnnIsInferring && !calledDnnInfer.current) {
      calledDnnInfer.current = true;
      inferOneEpoch();
    } else {
      calledDnnInfer.current = false;
    }
  }, [dnnIsInferring]);

  //
  // RESET LOGIC
  // On stop training & once at initialization time
  //
  useEffect(() => {
    if (!requestDnnTraining && !disabled) reset();
  }, [requestDnnTraining, disabled]);

  const reset = () => {
    // Training status
    setTrainingModelId(null);
    setPrevModelId(null);
    setTrainingEpoch(0);
    setTrainingTimeLeft(undefined);
    setRequestDnnTraining(false);
    setDnnIsTraining(false);

    // Inference status
    setInferenceEpoch(-1);
    setInferenceTimeLeft(undefined);
    setDnnIsInferring(false);

    // Autofill model name
    const name = username.slice(0, username.indexOf("@"));
    const date = dateFormat(new Date(), "mm-dd-yy_HH-MM");
    setModelName(`${name}_${date}`);

    // Clear form fields
    setDnnType(dnns[0].id);
    setDnnAugmentNegs(true);
    setDnnPosTags([]);
    setDnnNegTags([]);
    setDnnValPosTags([]);
    setDnnValNegTags([]);
    setDnnCheckpointModel(null);
  };

  //
  // MODEL INFO REFRESH
  //

  const updateModels = async () => {
    const url = new URL(`${endpoints.getModels}/${datasetName}`);
    const res = await fetch(url, {
      method: "GET",
      headers: {"Content-Type": "application/json"},
    }).then(res => res.json());
    setModelInfo(res.models);
  }

  // On training finish
  useEffect(() => {
    if (trainingEpoch > 0) updateModels();
  }, [trainingEpoch]);

  // On inference finish & once at initialization time
  useEffect(() => {
    if (inferenceJobId === null) updateModels();
  }, [inferenceJobId])

  const timeLeftToString = (t) => (t && t > 0) ? new Date(Math.max(t, 0) * 1000).toISOString().substr(11, 8) : "estimating...";

  const formatOptions = (d, idx) => {
    return (
      <FormGroup className="mx-1">
        <Label className="mb-0" htmlFor={"formatInput" + d.param}>{d.displayName}</Label>
        {(d.type === "switch" || d.type === "checkbox") ?
         <CustomInput
           id={"formatInput" + d.param}
           type={d.type}
           name={d.param}
           checked={dnnKwargs[d.param]}
           onChange={handleDnnKwargsChange}/> :
         <Input
           id={"formatInput" + d.param}
           type={d.type}
           name={d.param}
           value={dnnKwargs[d.param]}
           onChange={handleDnnKwargsChange}/>}
      </FormGroup>);
  };

  if (!isVisible) return null;
  return (
    <>
      <div className="d-flex flex-row align-items-center justify-content-between mb-1">
        {requestDnnTraining || dnnIsInferring ? <>
          <div className="d-flex flex-row align-items-center">
            <Spinner color="dark" className="my-1 mr-2" />
            {clusterReady ?
             <div>
               <div style={{verticalAlign: "top", display: "inline-block"}}>
                 Training model <b>{modelName} </b> &mdash;{" "}
               </div>
               <div style={{verticalAlign: "top", display: "inline-block"}}>
                 <div>
                   {requestDnnTraining ? "time left for training " + (dnnKwargs["epochs_to_run"] > 1 ? "epochs " + trainingEpoch + "-" + (trainingEpoch + dnnKwargs["epochs_to_run"]) : "epoch " + trainingEpoch) + ": " + timeLeftToString(trainingTimeLeft) : "training complete, waiting for inference"}
                 </div>
                 <div>
                   {dnnIsInferring && <span>time left for inference epoch {inferenceEpoch}: {timeLeftToString(inferenceTimeLeft)}</span>}
                 </div>
               </div>
               <br /> TensorBoard:{" "} {trainingTensorboardUrl ? <a href={trainingTensorboardUrl} target="_blank">link</a> : "loading..."}
             </div> :
             <b>Starting cluster</b>
            }
          </div>
          <Button
            color="danger"
            onClick={() => setRequestDnnTraining(false)}
          >Stop training</Button>
        </> : <>
          <FormGroup className="mb-0">
            <select className="custom-select mr-2" value={dnnType} onChange={e => setDnnType(e.target.value)}>
              {dnns.map((d) => <option key={d.id} value={d.id}>{d.label}</option>)}
            </select>
            <ReactSVG className="icon" src="assets/arrow-caret.svg" />
          </FormGroup>
          <Input
            className="mr-2"
            placeholder="Model name"
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
            spellcheck="false"
          />
          <CategoryInput
            id="dnn-pos-bar"
            className="mr-2"
            placeholder="Positive example tags"
            disabled={requestDnnTraining}
            customModesByCategory={customModesByCategory}
            selected={dnnPosTags}
            setSelected={setDnnPosTags}
          />
          <CategoryInput
            id="dnn-neg-bar"
            className="mr-2"
            placeholder="Negative example tags"
            disabled={requestDnnTraining}
            customModesByCategory={customModesByCategory}
            selected={dnnNegTags}
            setSelected={setDnnNegTags}
          />
          <div className="my-2 custom-control custom-checkbox">
            <input
              type="checkbox"
              className="custom-control-input"
              id="dnn-augment-negs-checkbox"
              disabled={requestDnnTraining}
              checked={dnnAugmentNegs}
              onChange={(e) => setDnnAugmentNegs(e.target.checked)}
            />
            <label className="custom-control-label text-nowrap mr-2" htmlFor="dnn-augment-negs-checkbox">
              Auto-augment negative set
            </label>
          </div>
          <Button
            color="light"
            onClick={() => setRequestDnnTraining(true)}
            disabled={disabled || dnnPosTags.length === 0 || (dnnNegTags.length === 0 && !dnnAugmentNegs)}
            >Start training
          </Button>
        </>}
      </div>
      {!requestDnnTraining && <div className="d-flex flex-row align-items-center mb-2">
        <FeatureInput
          id="checkpoint-model-bar"
          placeholder="Checkpoint to train from (optional)"
          features={modelInfo.filter(m => m.latest.has_checkpoint)}
          selected={dnnCheckpointModel}
          setSelected={setDnnCheckpointModel}
        />
        {dnnAugmentNegs && <>
          <CategoryInput
            id="dnn-augment-negs-include-bar"
            className="ml-2"
            placeholder="Tags to include in auto-negative pool"
            customModesByCategory={customModesByCategory}
            selected={dnnAugmentIncludeTags}
            setSelected={setDnnAugmentIncludeTags}
          />
          <CategoryInput
            id="dnn-augment-negs-exclude-bar"
            className="ml-2"
            placeholder="Tags to exclude from auto-negative pool"
            customModesByCategory={customModesByCategory}
            selected={dnnAugmentExcludeTags}
            setSelected={setDnnAugmentExcludeTags}
          />
        </>}
      </div>}
      {!requestDnnTraining && <a
        href="#"
        className="text-small text-muted"
        onClick={e => {
          setDnnAdvancedIsOpen(!dnnAdvancedIsOpen);
          e.preventDefault();
        }}
      >
        {dnnAdvancedIsOpen ? "Hide" : "Show"} advanced training options
      </a>}
      {!requestDnnTraining &&
      <Collapse isOpen={dnnAdvancedIsOpen && !requestDnnTraining} timeout={200}>
        <div>
          <CategoryInput
            id="dnn-val-pos-bar"
            className="mr-2"
            placeholder="Validation positive example tags"
            disabled={requestDnnTraining}
            customModesByCategory={customModesByCategory}
            selected={dnnValPosTags}
            setSelected={setDnnValPosTags}
          />
          <CategoryInput
            id="dnn-val-neg-bar"
            className="mr-2"
            placeholder="Validation negative example tags"
            disabled={requestDnnTraining}
            customModesByCategory={customModesByCategory}
            selected={dnnValNegTags}
            setSelected={setDnnValNegTags}
          />
          <div className="d-flex flex-row">
            {optInputs.map(formatOptions)}
          </div>
          <div className="d-flex flex-row my-1">
            {otherInputs.map(formatOptions)}
          </div>
        </div>
      </Collapse>}
    </>
  );
};

export default TrainPanel;
