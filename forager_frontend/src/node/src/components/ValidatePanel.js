import React, { useState, useEffect, useRef } from "react";
import {
  Button,
  Row,
  Col,
} from "reactstrap";

import fromPairs from "lodash/fromPairs";
import toPairs from "lodash/toPairs";
import isEmpty from "lodash/isEmpty";

import ActiveValidationModal from "./ActiveValidationModal";
import ImageStack from "./ImageStack";
import FeatureInput from "./FeatureInput";

const LABEL_STACK_SIZE = 10;

const endpoints = fromPairs(toPairs({
  queryMetrics: 'query_metrics',
  queryActiveValidation: 'query_active_validation',
  addValAnnotations: 'add_val_annotations',
}).map(([name, endpoint]) => [name, `${process.env.REACT_APP_SERVER_URL}/api/${endpoint}`]));

// https://reactjs.org/docs/hooks-faq.html
function usePrevious(value) {
  const ref = useRef();
  useEffect(() => {
    ref.current = value;
  });
  return ref.current;
}

const ValidatePanel = ({
  modelInfo,
  datasetName,
  datasetInfo,
  username,
  isVisible,
}) => {
  const [isValidating, setIsValidating] = useState(false);

  //
  // UI
  //
  const [validateModel, setValidateModel] = useState(null);
  const [modalIsOpen, setModalIsOpen] = useState(false);

  //
  // DATA CONNECTIONS
  //
  const [metrics, setMetrics] = useState({});
  const [metricsIsLoading, setMetricsIsLoading] = useState(false);
  const [activeVal, setActiveVal] = useState({});
  const [activeValIsLoading, setActiveValIsLoading] = useState(false);

  const [labelingStack, setLabelingStack] = useState([]);
  const [activeValIsStale, setActiveValIsStale] = useState(false);

  // Reload metrics on model changes
  const model = (isValidating && !!(validateModel)) ?
    modelInfo.find(m => m.name === validateModel.name).with_output : undefined;
  const modelId = (model || {}).model_id;
  const prevModelId = usePrevious(modelId);

  useEffect(() => {
    if (!modelId) return;
    if (modelId !== prevModelId) setActiveValIsStale(true);
    if (modelId && !isEmpty(activeVal.weights)) {
      setMetricsIsLoading(true);
    }
  }, [modelId, activeVal]);

  const getMetrics = async () => {
    const url = new URL(`${endpoints.queryMetrics}/${datasetName}`);
    let body = {
      model: modelId,
      index_id: datasetInfo.index_id,
      weights: activeVal.weights,
    };
    const res = await fetch(url, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    }).then(res => res.json());
    console.log(res);
    setMetrics(res);
  };

  const getActiveVal = async () => {
    const url = new URL(`${endpoints.queryActiveValidation}/${datasetName}`);
    let body = {
      model: modelId,
      index_id: datasetInfo.index_id,
      f1: metrics.f1,
    };
    const res = await fetch(url, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    }).then(res => res.json());
    getNewLabelStack(res);
    setMetricsIsLoading(true);
  };

  const submitLabels = async (labels) => {
    const url = new URL(endpoints.addValAnnotations);
    for (const ann of labels) {
      ann.mode = ann.value;
    }
    let body = {
      user: username,
      model: modelId,
      annotations: labels,
    };
    const res = await fetch(url, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    }).then(res => res.json());

    if (res.created !== labels.length) {
      console.warn("Incorrect number of annotations created by submitLabels", labels, res);
    }

    if (activeVal.paths.length === 0 || activeValIsStale) {
      setActiveValIsStale(false);
      setActiveValIsLoading(true);
    } else {
      getNewLabelStack(activeVal);
    }
  };

  const getNewLabelStack = (thisActiveVal) => {
    let paths = [...thisActiveVal.paths];
    let identifiers = [...thisActiveVal.identifiers];

    const stackPaths = paths.splice(0, LABEL_STACK_SIZE);
    const stackIdentifiers = identifiers.splice(0, LABEL_STACK_SIZE);

    const images = stackPaths.map((path, i) => {
      let filename = path.substring(path.lastIndexOf("/") + 1);
      let id = filename.substring(0, filename.indexOf("."));
      return {
        name: filename,
        src: path,
        id: stackIdentifiers[i],
        thumb: `https://storage.googleapis.com/foragerml/thumbnails/${datasetInfo.index_id}/${id}.jpg`,
      };
    });

    setLabelingStack(images);
    setActiveVal({...thisActiveVal, paths, identifiers});
  };

  useEffect(() => {
    if (metricsIsLoading) getMetrics().finally(() => setMetricsIsLoading(false));
  }, [metricsIsLoading]);

  useEffect(() => {
    if (activeValIsLoading) getActiveVal().finally(() => setActiveValIsLoading(false));
  }, [activeValIsLoading]);

  //
  // START & STOP VALIDATION
  //

  useEffect(() => {
    if (isValidating) {
      setActiveValIsLoading(true);
    } else if (!isValidating) {  // on validation stop + once on initial render
      setValidateModel(null);

      setMetrics({
        precision: null,
        precision_std: null,
        recall: null,
        recall_std: null,
        f1: null,
        f1_std: null,
        num_false_positives: null,
        num_false_negatives: null,
        num_labeled: null,
      });
      setMetricsIsLoading(false);

      setActiveVal({
        paths: [],
        identifiers: [],
        weights: {},
      });
      setActiveValIsLoading(false);

      setLabelingStack([]);
      setActiveValIsStale(false);
    }
  }, [isValidating]);

  //
  // LABELING
  //

  const startLabeling = () => {
    if (!!!(username)) return;
    setModalIsOpen(true);
  }

  if (!isVisible) return null;

  const isLoading = activeValIsLoading || metricsIsLoading;
  return (
    <>
      <ActiveValidationModal
        isOpen={modalIsOpen}
        setIsOpen={setModalIsOpen}
        images={labelingStack}
        model={model}
        submitLabels={submitLabels}
      />
      <div className={isLoading ? "loading" : ""}>
        <div className="d-flex flex-row align-items-center justify-content-between">
          <FeatureInput
            id="validate-model-bar"
            className="mr-2"
            placeholder="Model to validate"
            features={modelInfo.filter(m => ((m.with_output || {}).pos_tags || []).length > 0)}
            selected={validateModel}
            setSelected={setValidateModel}
            disabled={isValidating}
          />
          <Button
            color="light"
            onClick={() => setIsValidating(!isValidating)}
            disabled={isLoading || !!!(validateModel)}
          >{isValidating ? "Stop": "Start"} validation</Button>
        </div>
        {isValidating && <Row className="text-center mt-3 mb-1">
          <Col md="9" className="d-flex flex-column justify-content-center">
            <Row>
              <Col md="4">
                <div>
                  <span className="h2 mb-1">{metrics.f1 === null ? "?" : Number(metrics.f1).toFixed(2)}</span>
                  {metrics.f1_std !== null && <span className="h6">&nbsp;&plusmn; {Number(metrics.f1_std).toFixed(2)}</span>}
                </div>
                <h6>F1 score</h6>
              </Col>
              <Col md="4">
                <div>
                <span className="h2 mb-1">{metrics.precision === null ? "?" : Number(metrics.precision).toFixed(2)}</span>
                  {metrics.precision_std !== null && <span className="h6">&nbsp;&plusmn; {Number(metrics.precision_std).toFixed(2)}</span>}
                </div>
                <h6>Precision</h6>
              </Col>
              <Col md="4">
                <div>
                  <span className="h2 mb-1">{metrics.recall === null ? "?" : Number(metrics.recall).toFixed(2)}</span>
                  {metrics.recall_std !== null && <span className="h6">&nbsp;&plusmn; {Number(metrics.recall_std).toFixed(2)}</span>}
                </div>
                <h6>Recall</h6>
              </Col>
            </Row>
            <Row className="mt-2">
              {metrics.num_false_positives !== null && <Col md="4">
                <div>
                  <span className="h2 mb-1">{metrics.num_false_positives}</span>
                </div>
                <h6>False positives</h6>
              </Col>}
              {metrics.num_false_negatives !== null && <Col md="4">
                <div>
                  <span className="h2 mb-1">{metrics.num_false_negatives}</span>
                </div>
                <h6>False negatives</h6>
              </Col>}
              {metrics.num_labeled !== null && <Col md="4">
                <div>
                  <span className="h2 mb-1">{metrics.num_labeled}</span>
                </div>
                <h6>Labeled images</h6>
              </Col>}
            </Row>
          </Col>
          <Col md="3">
            {labelingStack.length > 0 && <ImageStack
              images={labelingStack}
              onClick={startLabeling}
              labelText="(Click to start labeling)"
            />}
          </Col>
        </Row>}
      </div>
    </>
  );
};

export default ValidatePanel;
