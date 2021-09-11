import React, { useState, useEffect } from "react";
import { InputGroup, InputGroupText, InputGroupAddon, Input } from 'reactstrap';
import { Button, Container,
         Popover,
         PopoverHeader,
         PopoverBody,
         Form,
         FormGroup,
         Label
} from "reactstrap";
import BounceLoader from "react-spinners/BounceLoader";
import { Link } from "react-router-dom";
import { faPlus } from "@fortawesome/free-solid-svg-icons";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";

import fromPairs from "lodash/fromPairs";
import toPairs from "lodash/toPairs";

const endpoints = fromPairs(toPairs({
  getDatasets: "get_datasets",
  createDataset: "create_dataset",
}).map(([name, endpoint]) => [name, `${process.env.REACT_APP_SERVER_URL}/api/${endpoint}`]));

const DatasetList = () => {
  const [datasets, setDatasets] = useState([]);

  async function getDatasetList() {
    const url = new URL(endpoints.getDatasets);
    let _datasets = await fetch(url, {
      method: "GET",
    }).then(r => r.json());
    setDatasets(_datasets.dataset_names);
  }

  useEffect(() => {
    getDatasetList();
  }, []);

  const [createDatasetPopoverOpen, setCreateDatasetPopoverOpen] = useState(false);
  const [createDatasetName, setCreateDatasetName] = useState("");
  const [createDatasetTrainDirectory, setCreateDatasetTrainDirectory] = useState("");
  const [createDatasetValDirectory, setCreateDatasetValDirectory] = useState("");
  const [createDatasetLoading, setCreateDatasetLoading] = useState(false);
  const toggleCreateDatasetPopoverOpen = () => {
    setCreateDatasetPopoverOpen((value) => !value);
  }
  const createDataset = (e) => {
    e.preventDefault();
    const fn = async () => {
      const url = new URL(endpoints.createDataset);
      const body = {
        dataset: e.target.datasetName.value,
        train_images_directory: e.target.trainDirectoryPath.value,
        val_images_directory: e.target.valDirectoryPath.value,
      };
      setCreateDatasetLoading(true);
      let resp = await fetch(url, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body),
      }).then(r => r.json());
      setCreateDatasetLoading(false);
      if (resp.status !== "success") {
        console.log(resp)
      } else {
        getDatasetList();
        setCreateDatasetPopoverOpen(false);
      }
    };
    fn();
  };

  return (
    <Container>
      <h2>Forager</h2>
      <div>
        <h3>Datasets</h3>
        {datasets.length > 0
                         ? datasets.map(d => <div>
                           <Link to={`/${d}`} activeClassName="active">{d}</Link>
                         </div>)
                         : <p>No datasets available.</p>}
        <div className="create-dataset-container">
          <Button
            id="create-dataset-open-button"
            color="light"
            size="md"
            type="button"
          >
            <FontAwesomeIcon
              icon={faPlus}
              className="mr-1"
            />
            Add dataset
          </Button>
          <Popover
            placement="right-start"
            target="create-dataset-open-button"
            isOpen={createDatasetPopoverOpen}
            toggle={toggleCreateDatasetPopoverOpen}
            trigger="legacy" >
            <PopoverHeader>Create Dataset</PopoverHeader>
            <PopoverBody>
              <Form onSubmit={createDataset}>
                <FormGroup>
                  <Label for="datasetNameInput">Dataset Name</Label>
                  <Input type="text" name="datasetName" id="datasetNameInput" />
                </FormGroup>
                <FormGroup>
                  <Label for="trainDirectory">Train Images Directory</Label>
                  <Input type="text" name="trainDirectoryPath" id="trainDirectory" />
                </FormGroup>
                <FormGroup>
                  <Label for="valDirectory" >Validation Images Directory</Label>
                  <Input type="text" name="valDirectoryPath" id="valDirectory" />
                </FormGroup>
                <Button size="sm" type="submit" disabled={createDatasetLoading}>Create </Button>
                <BounceLoader css={{position: "absolute", marginLeft: "2em"}} size={30} loading={createDatasetLoading} color="purple" />
              </Form>
            </PopoverBody>
          </Popover>
        </div>
      </div>
    </Container>
  );
};

export default DatasetList;
