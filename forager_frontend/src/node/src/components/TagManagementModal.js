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

import fromPairs from "lodash/fromPairs";
import sortBy from "lodash/sortBy";
import toPairs from "lodash/toPairs";

import { ConfirmModal } from "../components";

const endpoints = fromPairs(toPairs({
  updateCategory: 'update_category',
  deleteCategory: 'delete_category',
}).map(([name, endpoint]) => [name, `${process.env.REACT_APP_SERVER_URL}/api/${endpoint}`]));

// TODO(mihirg): Combine with this same constant in other places
const BUILT_IN_MODES = [
  ["POSITIVE", "Positive"],
  ["NEGATIVE", "Negative"],
  ["HARD_NEGATIVE", "Hard Negative"],
  ["UNSURE", "Unsure"],
];

const TableContainer = styled.div`
  height: 70vh;
  overflow: auto;
`;

const TagManagementModal = ({
  isOpen,
  toggle,
  datasetName,
  categoryCounts,
  categoryDispatch,
  username,
  isReadOnly
}) => {
  // Aggregate all non-built-in modes into "other" count
  const displayCounts = useMemo(() => {
    let result = {};
    for (const [category, counts] of Object.entries(categoryCounts)) {
      let innerResult = {};
      innerResult["OTHER"] = 0
      for (const [mode, count] of Object.entries(counts)) {
        if (BUILT_IN_MODES.some(([m]) => m === mode)) {
          innerResult[mode] = count;
        } else {
          innerResult["OTHER"] += count;
        }
      }
      result[category] = innerResult;
    }
    return result;
  }, [categoryCounts]);
  const categoryList = useMemo(() => sortBy(Object.keys(displayCounts), c => c.toLowerCase()), [displayCounts]);

  const [categories, setCategories] = useState([]);

  const [orderBy, setOrderBy_] = useState("name");
  const [hideOthersTags, setHideOthersTags] = useState(false);
  const [hideIngestTags, setHideIngestTags] = useState(false);

  const [orderAscending, setOrderAscending] = useState(true);

  const [confirmIsOpen, setConfirmIsOpen] = useState(false);
  const [confirmCategory, setConfirmCategory] = useState(null);
  const [confirmCategoryIdx, setConfirmCategoryIdx] = useState(null);
  const toggleConfirmIsOpen = (category) => setConfirmIsOpen(!confirmIsOpen);

  const sortCategories = (arr) => {
    const copy = arr.slice(0);
    const m = orderAscending ? 1 : -1;
    if (orderBy === "name") {
      copy.sort((a, b) => m * (a.tag.toLowerCase() < b.tag.toLowerCase() ? -1 : 1));
    } else {
      copy.sort((a, b) => m * (displayCounts[a.tag][orderBy] - displayCounts[b.tag][orderBy]));
    }
    return copy;
  };

  const setOrderBy = (by) => {
    if (by === orderBy) {
      setOrderAscending(!orderAscending);
    } else {
      setOrderBy_(by);
      setOrderAscending(true);
    }
  };

  const setCategoryByIndex = (tag, idx) => {
    if (isReadOnly) return;

    const oldTag = categories[idx].tag;

    categories[idx].tag = tag;
    setCategories(categories.slice(0));
  };

  const updateCategoryByIndex = async (e, idx, srcIdx) => {
    if (isReadOnly) return;

    const newTag = categories[idx].tag;
    const oldTag = categoryList[srcIdx];
    if (newTag === oldTag) return;

    if (categoryList.find((name) => name === newTag) !== undefined) {
      e.target.setCustomValidity(`"${newTag}" already exists`);
      e.target.reportValidity();
      categories[idx].tag = oldTag;
      setCategories(categories.slice(0));
      return;
    }

    const url = new URL(endpoints.updateCategory);
    const body = {
      user: username,
      newCategory: newTag,
      oldCategory: oldTag,
    };

    const res = await fetch(url, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    }).then(res => res.json());

    categoryDispatch({
      type: "RENAME_CATEGORY",
      newCategory: newTag,
      oldCategory: oldTag,
    });
  };

  const deleteCategoryByIndex = async (idx) => {
    if (isReadOnly) return;

    const tag = categories[idx].tag;
    const url = new URL(endpoints.deleteCategory);
    const body = {
      user: username,
      category: tag,
    };

    const res = await fetch(url, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    }).then(res => res.json());

    categoryDispatch({
      type: "DELETE_CATEGORY",
      category: tag,
    });
  };

  useEffect(() => {
    setCategories(sortCategories(categoryList.map(
      (tag, i) => ({tag, srcIdx: i})
    )));
  }, [JSON.stringify(categoryList)]);

  useEffect(() => {
    setCategories(prev => sortCategories(prev))
  }, [categoryCounts, orderBy, orderAscending]);

  /* useEffect(async () => {
   *   const url = new URL(endpoints.getCategoryCounts + `/${datasetName}`);
   *   const body = {
   *     user: username,
   *   };
   *   url.search = new URLSearchParams(body).toString();
   *   const res = await fetch(url, {
   *     method: "GET",
   *   }).then(res => res.json());

   *   let cats = Object.fromEntries(Object.entries(res).map(([key, val]) => [key, Object.keys(val)]))

   *   if (hideOthersTags) {
   *   }
   *   if (hideIngestTags) {
   *   }
   *   setCategories(prev => sortCategories(prev))
   * }, [hideOthersTags, hideIngestTags]);
   */
  const tableBodyFromTags = () => {
    return (
      <tbody>
        {categories.map((obj, i) => {
          const counts = displayCounts[categoryList[obj.srcIdx]] || {};
          return (
            <tr key={obj.srcIdx}>
              <td>
                <input
                  type="text"
                  value={obj.tag}
                  disabled={isReadOnly}
                  onChange={(e) => setCategoryByIndex(e.target.value, i)}
                  onKeyDown={(e) => { if (e.keyCode === 13) e.target.blur(); }}
                  onBlur={(e) => updateCategoryByIndex(e, i, obj.srcIdx)}
                />
              </td>
              {BUILT_IN_MODES.map(([mode]) => <td>{counts[mode] || 0}</td>)}
              <td>{counts["OTHER"]}</td>
              <td>
                <Button close disabled={isReadOnly} onClick={(e) => {
                    setConfirmCategory(obj.tag);
                    setConfirmCategoryIdx(i);
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
        Manage Tags
      </ModalHeader>
      <ModalBody>
        <div className="custom-switch custom-control mr-4">
          <Input type="checkbox" className="custom-control-input"
            id="hide-others-tags-switch"
            checked={hideOthersTags}
            onChange={(e) => setHideOthersTags(e.target.checked)}
          />
          <label className="custom-control-label text-nowrap" htmlFor="hide-others-tags-switch">
            Hide tags from others
          </label>
        </div>
        <div className="custom-switch custom-control mr-4">
          <Input type="checkbox" className="custom-control-input"
            id="hide-ingest-tags-switch"
            checked={hideIngestTags}
            onChange={(e) => setHideIngestTags(e.target.checked)}
          />
          <label className="custom-control-label text-nowrap" htmlFor="hide-ingest-tags-switch">
            Hide tags from ingest
          </label>
        </div>
        <TableContainer className="pr-2">
          <Table hover borderless size="sm">
            <thead>
              <tr>
                <th style={{cursor: "pointer"}} onClick={() => setOrderBy("name")}>
                  Tag Name <FontAwesomeIcon icon={
                    orderBy !== "name" ? faSort : (orderAscending ? faSortUp : faSortDown)
                  } />
                </th>
                {BUILT_IN_MODES.concat([["OTHER", "Other"]]).map(([mode, name]) => (
                  <th style={{cursor: "pointer"}} onClick={() => setOrderBy(mode)}>
                    {name} <FontAwesomeIcon icon={
                      orderBy !== mode ? faSort : (orderAscending ? faSortUp : faSortDown)
                    } />
                  </th>
                ))}
              </tr>
            </thead>
            {tableBodyFromTags()}
          </Table>
        </TableContainer>

        <ConfirmModal
          isOpen={confirmIsOpen}
          toggle={toggleConfirmIsOpen}
          message={<span>Are you sure you want to delete the tag <strong>{confirmCategory}</strong>? This action cannot be undone.</span>}
          confirmBtn={<Button color="danger" onClick={(e) => {
            deleteCategoryByIndex(confirmCategoryIdx);
            toggleConfirmIsOpen();
          }}>Delete</Button>}
          cancelBtn={<Button color="light" onClick={(e) => toggleConfirmIsOpen()}>Cancel</Button>}
        />
      </ModalBody>
    </Modal>
  );
};

export default TagManagementModal;
