import React, { useState, useEffect } from "react";
import {
  Modal,
  ModalHeader,
  ModalFooter,
  ModalBody,
  Form,
  FormGroup,
  Input,
  Label,
  FormText,
  Button,
} from "reactstrap";
import { Range } from "rc-slider";

import fromPairs from "lodash/fromPairs";
import toPairs from "lodash/toPairs";

// TODO(mihirg): Combine with this same constant in other places
const LABEL_VALUES = [
  ["POSITIVE", "Positive"],
  ["NEGATIVE", "Negative"],
  ["HARD_NEGATIVE", "Hard Negative"],
  ["UNSURE", "Unsure"],
];

const endpoints = fromPairs(toPairs({
  addAnnotationsToResultSet: 'add_annotations_to_result_set',
}).map(([name, endpoint]) => [name, `${process.env.REACT_APP_SERVER_URL}/api/${endpoint}`]));

const BulkTagModal = ({
  isOpen,
  toggle,
  resultSet,
  customModesByCategory,
  categoryDispatch,
  username,
}) => {
  const [category, setCategory] = useState("");
  const [selectedValue, setSelectedValue] = useState();
  const [selectedRange, setSelectedRange] = useState([0, 0]);
  const [isLoading, setIsLoading] = useState(false);

  const trimmedCategory = category.trim();
  const numImages = Math.floor(resultSet.num_results * (selectedRange[1] - selectedRange[0]) / 100);
  const isNew = !customModesByCategory.has(trimmedCategory);

  // Default values (clear every time form is opened)
  useEffect(() => {
    if (isOpen) {
      setCategory("");
      setSelectedValue(LABEL_VALUES[0][0]);
      setSelectedRange([0, 100]);
    }
  }, [isOpen]);

  const apply = async () => {
    const url = new URL(endpoints.addAnnotationsToResultSet);
    const body = {
      user: username,
      category: trimmedCategory,
      mode: selectedValue,
      result_set_id: resultSet.id,
      from: selectedRange[0] / 100,
      to: selectedRange[1] / 100,
    };

    const res = await fetch(url, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    }).then(res => res.json());

    if (res.created !== numImages) {
      console.warn(`Bulk tag expected to create ${numImages}, actually created ${res.created}`);
    }
    if (res.created > 0) {
      if (isNew) {
        categoryDispatch({
          type: "ADD_CATEGORY",
          category: trimmedCategory,
        })
      }
      toggle();
    }
  };

  useEffect(() => {
    if (isLoading) apply().finally(() => setIsLoading(false));
  }, [isLoading]);

  return (
    <Modal
      isOpen={isOpen}
      toggle={toggle}
      modalTransition={{ timeout: 25 }}
      backdropTransition={{ timeout: 75 }}
      contentClassName={isLoading ? "loading" : ""}
    >
      <ModalHeader>Bulk tag results</ModalHeader>
      <ModalBody>
        <Form>
          <FormGroup>
            <Input
              placeholder="Category name"
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              spellcheck="false"
              disabled={isLoading}
            />
            <FormText className="font-weight-normal" color={isNew ? "muted" : "danger"}>
              We suggest you use a new category name so that this operation is easy to undo if necessary.
            </FormText>
          </FormGroup>
          <FormGroup>
            {LABEL_VALUES.map(([value, name]) =>
              <span>
                <a
                  href="#"
                  onClick={(e) => {
                    if (isLoading) return;
                    setSelectedValue(value);
                    e.preventDefault();
                  }}
                  className={`rbt-token ${value} ${selectedValue === value ? "rbt-token-active" : ""}`}
                >
                  {name.toLowerCase()}
                </a>
              </span>)
            }
          </FormGroup>
          <div className="mb-1">
            ...will be applied to range <b>{selectedRange[0]}%</b> to <b>{selectedRange[1]}%</b>
            <span className="text-muted"> ({numImages} image{numImages === 1 ? "" : "s"}) </span>
            of results
          </div>
          <div className="px-1">
            <Range
              allowCross={false}
              value={selectedRange}
              step={5}
              dots
              onChange={setSelectedRange}
              disabled={isLoading}
            />
          </div>
        </Form>
      </ModalBody>
      <ModalFooter>
        <Button
          color="light"
          onClick={toggle}
          disabled={isLoading}
        >Cancel</Button>{" "}
        <Button
          color="primary"
          disabled={!!!(trimmedCategory) || numImages === 0 || isLoading}
          onClick={() => setIsLoading(true)}
        >Apply</Button>
      </ModalFooter>
    </Modal>
  );
};

export default BulkTagModal;
