import fromPairs from "lodash/fromPairs";
import toPairs from "lodash/toPairs";

export const WEB_SERVER_URL = `http://${window.location.hostname.toString()}:${
  process.env.REACT_APP_SERVER_PORT
}`;

export const endpoints = fromPairs(
  toPairs({
    queryMetrics: "query_metrics",
    queryActiveValidation: "query_active_validation",
    addValAnnotations: "add_val_annotations",

    trainModel: "train_model",
    modelStatus: "model",
    modelInference: "model_inference",
    modelInferenceStatus: "model_inference_status",
    stopModelInference: "stop_model_inference",
    startCluster: "start_cluster",
    clusterStatus: "cluster",

    updateCategory: "update_category",
    deleteCategory: "delete_category",

    deleteModelOutput: "delete_model_output",
    getModelOutputs: "get_model_outputs",

    updateModel: "update_model",
    deleteModel: "delete_model",
    getModels: "get_models",

    generateTextEmbedding: "generate_text_embedding",
    addAnnotationsToResultSet: "add_annotations_to_result_set",

    getDatasetInfo: "get_dataset_info",
    getCategoryCounts: "get_category_counts",
    getResults: "get_results",
    trainSvm: "train_svm",
    queryImages: "query_images",
    querySvm: "query_svm",
    queryKnn: "query_knn",
    queryRanking: "query_ranking",
    generateEmbedding: "generate_embedding",
    keepAlive: "keep_alive",

    getDatasets: "get_datasets",
    createDataset: "create_dataset",
    deleteDataset: "delete_dataset",
    startInference: "start_model_inference",
    inferenceStatus: "model_inference_status",

    getAnnotations: "get_annotations",
    addAnnotations: "add_annotations",
  }).map(([name, endpoint]) => [name, `${WEB_SERVER_URL}/api/${endpoint}`])
);
