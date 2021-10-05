export type DatasetInfo = {
    isNotLoaded: boolean;
    index_id: boolean;
    name: string;
    created_at: string;
    created_by: string;
    status: string;
};


enum ClusterConnectionStatus {
    Creating = "creating",
    Ready = "ready",
    Destroying = "destroying",
}

export type ClusterInfo = {
    id: string;
    name: string;
    created_at: string;
    created_by: string;
    status: ClusterConnectionStatus;
};
