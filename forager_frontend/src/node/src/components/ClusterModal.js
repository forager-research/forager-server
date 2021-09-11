import React, { useState, useCallback, useEffect, useRef } from "react";
import {
  Button,
  Form,
  FormGroup,
  Modal,
  ModalHeader,
  ModalBody,
} from "reactstrap";
import { faMousePointer } from "@fortawesome/free-solid-svg-icons";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";

import fromPairs from "lodash/fromPairs";
import toPairs from "lodash/toPairs";
import differenceWith from "lodash/differenceWith";
import intersectionWith from "lodash/intersectionWith";
import isEqual from "lodash/isEqual";
import unionWith from "lodash/unionWith";
import uniqWith from "lodash/uniqWith";

import ImageGrid from "./ImageGrid";
import NewModeInput from "./NewModeInput";
import CategoryInput from "./CategoryInput";
import AnnotatedImage from "./AnnotatedImage";

const endpoints = fromPairs(toPairs({
  getAnnotations: 'get_annotations',
  addAnnotations: 'add_annotations',
}).map(([name, endpoint]) => [name, `${process.env.REACT_APP_SERVER_URL}/api/${endpoint}`]));

const imageGridSizes = [
  {label: "small", size: 125},
  {label: "medium", size: 250},
  {label: "large", size: 375},
];

// TODO(mihirg): Combine with this same constant in other places
const BUILT_IN_MODES = [
  ["POSITIVE", "Positive"],
  ["NEGATIVE", "Negative"],
  ["HARD_NEGATIVE", "Hard Negative"],
  ["UNSURE", "Unsure"],
];

const ClusterModal = ({
  isOpen,
  setIsOpen,
  isImageOnly,
  isReadOnly,
  selection,
  setSelection,
  clusters,
  findSimilar,
  customModesByCategory,
  categoryDispatch,
  username,
  setSubset,
  labelCategory,
  mode,
}) => {
  const typeaheadRef = useRef();

  const selectedCluster = (selection.cluster !== undefined &&
                           selection.cluster < clusters.length) ?
                          clusters[selection.cluster] : undefined;
  const isSingletonCluster = (selectedCluster !== undefined &&
                              selectedCluster.length === 1);

  let selectedImage;
  if (isSingletonCluster) {
    selectedImage = (selectedCluster !== undefined) ? selectedCluster[0] : undefined;
  } else {
    selectedImage = (selectedCluster !== undefined &&
                     selection.image !== undefined &&
                     selection.image < selectedCluster.length) ?
                    selectedCluster[selection.image] : undefined;
  }
  const isClusterView = (selectedCluster !== undefined &&
                         selectedImage === undefined);
  const isImageView = !isSingletonCluster && !isClusterView;

  //
  // DATA CONNECTIONS
  //

  const [annotations, setAnnotations] = useState({});
  const [showBoxes, setShowBoxes] = useState(false);

  const toggleShowBoxes = () => {
    setShowBoxes(prev => !prev);
  };

  // Reload annotations whenever there's a new result set
  useEffect(() => {
    const fn = async () => {
      if (clusters.length === 0) return;
      let annotationsUrl = new URL(endpoints.getAnnotations);
      let body = {
        identifiers: clusters.map(cl => cl.map(im => im.id)).flat()
      }
      setAnnotations(await fetch(annotationsUrl, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body),
      }).then(r => r.json()));
    };
    fn();
  }, [clusters]);

  //
  // IMAGE SELECTION
  //

  const [excludedImageIndexes, setExcludedImageIndexes] = useState({});
  const [imageGridSize, setImageGridSize_] = useState(imageGridSizes[0]);

  useEffect(() => {
    setExcludedImageIndexes({});  // Select all
  }, [selectedCluster]);

  const handleGalleryClick = (e, i) => {
    if (e.shiftKey) {
      toggleImageSelection(i);
    } else {
      setSelection({
        cluster: selection.cluster,
        image: i
      });
    }
  };

  const toggleImageSelection = (i, e) => {
    let newExcludedImageIndexes = {...excludedImageIndexes};
    newExcludedImageIndexes[i] = !!!(newExcludedImageIndexes[i]);
    setExcludedImageIndexes(newExcludedImageIndexes);
    if (e) e.preventDefault();
  }

  const setImageGridSize = (size, e) => {
    setImageGridSize_(size);
    e.preventDefault();
  };

  //
  // TAGGING
  //

  const [isLoading, setIsLoading] = useState(false);

  const getImageTags = im => ((annotations[im.id] && annotations[im.id]['tags']) || []);
  const getImageBoxes = im => ((annotations[im.id] && annotations[im.id]['boxes']) || []);
  let selectedTags = [];
  let selectedBoxes = [];
  if (selectedImage !== undefined) {
    selectedTags = getImageTags(selectedImage);
    selectedBoxes = getImageBoxes(selectedImage);
  } else if (selectedCluster !== undefined) {
    selectedTags = intersectionWith(...(selectedCluster.flatMap((im, i) =>
      excludedImageIndexes[i] ? [] : [getImageTags(im)])), isEqual);
  }

  // Add or remove tags whenever the typeahead value changes
  const addAnnotations = async (category, value, identifiers) => {
    const url = new URL(endpoints.addAnnotations);
    const body = {
      user: username,
      category: category,
      mode: value,
      identifiers: identifiers
    };
    return fetch(url, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    });
  }
  const onTagsChanged = async (newTags) => {
    const added = differenceWith(newTags, selectedTags, isEqual);
    const deleted = differenceWith(selectedTags, newTags, isEqual);
    const deletedNotChanged = differenceWith(deleted, added, (a, b) => a.category === b.category);
    const imageIds = (selectedImage !== undefined) ? [selectedImage.id] :
      selectedCluster.flatMap((im, i) => excludedImageIndexes[i] ? [] : [im.id]);

    let newAnnotations = {...annotations};
    for (const id of imageIds) {
      const minusDeleted = differenceWith(annotations[id] && annotations[id]['tags'] || [], deleted, isEqual);
      let plusAdded = unionWith(minusDeleted, added, isEqual);
      let deduplicated = uniqWith(plusAdded.reverse(), (a, b) => a.category === b.category);
      if (!(id in newAnnotations)) {
        newAnnotations[id] = {'tags': [], 'boxes': []}
      }
      newAnnotations[id]['tags'] = deduplicated.reverse();
    }
    setIsLoading(true);
    setAnnotations(newAnnotations);

    let addPromises = added.map(async t => addAnnotations(t.category, t.value, imageIds));
    let deletePromises = deletedNotChanged.map(async t => addAnnotations(t.category, "TOMBSTONE", imageIds));
    await Promise.all([...addPromises, ...deletePromises]);

    setIsLoading(false);
  }

  //
  // KEY BINDINGS
  //

  const handleKeyDown = useCallback((e) => {
    const { key } = e;
    const keyAsNumber = parseInt(key);

    let caught = true;
    if (isClusterView && key === "ArrowDown") {
      // Switch to image view
      setSelection({
        cluster: selection.cluster,
        image: 0
      });
    } else if (isImageView && key === "ArrowUp") {
      // Switch to cluster view
      setSelection({
        cluster: selection.cluster,
      });
    } else if (isImageView && key === "ArrowLeft") {
      // Previous image
      setSelection({
        cluster: selection.cluster,
        image: Math.max(selection.image - 1, 0)
      });
    } else if (isImageView && key === "ArrowRight") {
      // Next image
      setSelection({
        cluster: selection.cluster,
        image: Math.min(selection.image + 1, clusters[selection.cluster].length - 1)
      });
    } else if (key === "ArrowLeft") {
      // Previous cluster
      setSelection({
        cluster: Math.max(selection.cluster - 1, 0),
        image: selection.image && 0
      });
    } else if (key === "ArrowRight") {
      // Next cluster
      setSelection({
        cluster: Math.min(selection.cluster + 1, clusters.length - 1),
      });
    } else if (isImageView && key === "s") {
      // Toggle selection
      toggleImageSelection(selection.image);
    } else if (key === "ArrowUp") {
      // Close modal
      setIsOpen(false);
    } else if (!isReadOnly && mode === "label" && !!(labelCategory) && !isNaN(keyAsNumber)) {
      // Label mode
      let boundValue;
      if (keyAsNumber >= 1 && keyAsNumber <= BUILT_IN_MODES.length) {
        boundValue = BUILT_IN_MODES[keyAsNumber - 1][0];
        caught = true;
      } else {
        const customIndex = (keyAsNumber === 0 ? 10 : keyAsNumber) - BUILT_IN_MODES.length - 1;
        boundValue = (customModesByCategory.get(labelCategory) || [])[customIndex];
      }
      if (boundValue !== undefined) {
        onTagsChanged([...selectedTags, {category: labelCategory, value: boundValue}]);
      } else {
        caught = false;
      }
    } else if (isClusterView && key === "s") {
      // Select all
      setExcludedImageIndexes({});
    } else if (isClusterView && key === "i") {
      // Invert selection
      setExcludedImageIndexes(fromPairs(selectedCluster.flatMap((_, i) =>
        !!!(excludedImageIndexes[i]) ? [[i, true]] : [])));
    } else if (isClusterView && key === "d") {
      // Deselect all
      setExcludedImageIndexes(fromPairs(selectedCluster.map((_, i) => [i, true])));
    } else if (key === "b") {
      toggleShowBoxes();
    } else if (key === "ArrowDown") { } else {
      caught = false;
    }
    if (caught) {
      e.preventDefault();
      typeaheadRef.current.blur();
      typeaheadRef.current.hideMenu();
    }
  }, [isClusterView, isImageView, clusters, selection, setSelection, typeaheadRef, excludedImageIndexes, selectedTags, annotations]);

  const handleTypeaheadKeyDown = (e) => {
    const { key } = e;
    if (key === "s" || key === "i" || key === "d" || key === "b") {
      e.stopPropagation();
    }
  }

  useEffect(() => {
    if (!isOpen) return;
    document.addEventListener("keydown", handleKeyDown)
    return () => {
      document.removeEventListener("keydown", handleKeyDown)
    }
  }, [isOpen, handleKeyDown]);

  //
  // RENDERING
  //

  let header;
  if (selectedCluster !== undefined) {
    header = `${isImageOnly ? "Image" : "Cluster"} ${selection.cluster + 1} of ${clusters.length}`;
    if (isClusterView) {
      header += ` (${selectedCluster.length} images)`;
    } else if (isSingletonCluster && !isImageOnly) {
      header += " (1 image)";
    } else if (isImageView) {
      header += `, image ${selection.image + 1} of ${clusters[selection.cluster].length}`;
    }
  }

  return (
    <Modal
      isOpen={isOpen}
      toggle={() => setIsOpen(false)}
      modalTransition={{ timeout: 25 }}
      backdropTransition={{ timeout: 75 }}
      size="full"
      className={`cluster-modal ${isLoading ? "loading" : ""}`}
    >
      {(selectedCluster !== undefined) && <>
        <ModalHeader toggle={() => setIsOpen(false)}>
          <span>{header}</span>
        </ModalHeader>

        <ModalBody>
          <p>
            <b>Key bindings:</b> &nbsp;
            <kbd>&larr;</kbd> <kbd>&rarr;</kbd> to move between {(isImageView || isImageOnly) ? "images" : "clusters"}
            {isClusterView && <>,{" "}
              <kbd>&darr;</kbd> or <FontAwesomeIcon icon={faMousePointer} /> to go into image view,{" "}
              <kbd>&uarr;</kbd> to go back to query results,{" "}
              <kbd>shift</kbd> <FontAwesomeIcon icon={faMousePointer} /> to toggle image selection,{" "}
              <kbd>s</kbd> to select all, <kbd>i</kbd> to invert selection, <kbd>d</kbd> to deselect all</>}
            {isImageView && <>,{" "}
              <kbd>&uarr;</kbd> to go back to cluster view,{" "}
              <kbd>s</kbd> or <FontAwesomeIcon icon={faMousePointer} /> to toggle image selection</>}
            {isSingletonCluster && <>,{" "}
              <kbd>&uarr;</kbd> to go back to query results</>}
            {" "}<kbd>b</kbd> to show/hide bounding boxes
          </p>
          {mode === "label" && (labelCategory ? <p>
            <b>Label mode:</b> &nbsp;
            {BUILT_IN_MODES.map(([value], i) =>
              <>
                <kbd>{i + 1}</kbd> <span className={`rbt-token ${value}`}>{labelCategory}</span>{" "}
              </>)}
            {(customModesByCategory.get(labelCategory) || []).map((name, i) =>
              <>
                <kbd>{(BUILT_IN_MODES.length + i + 1) % 10}</kbd> <span className="rbt-token CUSTOM">{labelCategory} ({name})</span>{" "}
              </>)}
            <NewModeInput
              category={labelCategory}
              customModesByCategory={customModesByCategory}
              categoryDispatch={categoryDispatch}
            />
          </p> : <p><b>Label mode:</b> No category specified</p>)}
          <Form>
            <FormGroup className="d-flex flex-row align-items-center mb-2">
              {/* TODO(mihirg): Change tags -> categories in code for consistency of terminology */}
              <CategoryInput
                allowNew
                id="image-tag-bar"
                placeholder="Image tags"
                disabled={isReadOnly || (isClusterView && selectedCluster.length === Object.values(excludedImageIndexes).filter(Boolean).length)}
                customModesByCategory={customModesByCategory}
                categoryDispatch={categoryDispatch}
                selected={selectedTags}
                setSelected={onTagsChanged}
                innerRef={typeaheadRef}
                onBlur={() => typeaheadRef.current.hideMenu()}
                onKeyDown={handleTypeaheadKeyDown}
                deduplicateByCategory
                allowNewModes
              />
              {(isClusterView) ?
                <Button color="light" className="ml-2" onClick={() => setSubset(selectedCluster)}>
                  Descend into cluster
                </Button> :
                <Button color="warning" className="ml-2" onClick={() => findSimilar(selectedImage)}>
                  Find similar images
                </Button>}
            </FormGroup>
          </Form>
          {selectedImage !== undefined ?
           <>{isImageView ?
              <a href="#" onClick={(e) => toggleImageSelection(selection.image, e)} className="selectable-image">
                <div className={"image " + (
                  !!!(excludedImageIndexes[selection.image]) ? "selected" : "")}>
                  <AnnotatedImage url={selectedImage.src} boxes={showBoxes ? selectedBoxes : []}/>
                </div>
              </a>
             :
              <AnnotatedImage url={selectedImage.src} boxes={showBoxes ? selectedBoxes : []}/>
           }</>
          :
           <>
             <div className="mb-1 text-small text-secondary font-weight-normal">
               Selected {selectedCluster.length - Object.values(excludedImageIndexes).filter(Boolean).length}{" "}
               of {selectedCluster.length} images (thumbnails: {imageGridSizes.map((size, i) =>
                 <>
                   <a key={i} href="#" className="text-secondary" onClick={(e) => setImageGridSize(size, e)}>{size.label}</a>
                   {(i < imageGridSizes.length - 1) ? ", " : ""}
                 </>
                 )})
             </div>
             <ImageGrid
               images={selectedCluster}
               annotations={showBoxes ? annotations : {}}
               onClick={handleGalleryClick}
               selectedPred={i => !!!(excludedImageIndexes[i])}
               minRowHeight={imageGridSize.size}
               imageAspectRatio={3/2}
             />
           </>
          }
        </ModalBody>
      </>}
    </Modal>
  );
}

export default ClusterModal;
