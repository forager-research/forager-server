import React, { useState, useCallback, useEffect } from "react";
import {
  Button,
  Form,
  FormGroup,
  Modal,
  ModalHeader,
  ModalBody,
  ModalFooter,
} from "reactstrap";

import isEqual from "lodash/isEqual";
import remove from "lodash/remove";
import uniq from "lodash/uniq";

// TODO(mihirg): Combine with this same constant in other places
const BUILT_IN_MODES = [
  ["POSITIVE", "Positive"],
  ["NEGATIVE", "Negative"],
  ["HARD_NEGATIVE", "Hard Negative"],
  ["UNSURE", "Unsure"],
];

const ActiveValidationModal = ({
  isOpen,
  setIsOpen,
  images,
  model,
  submitLabels,
}) => {
  const [isSubmitting, setIsSubmitting] = useState(false);

  const [index, setIndex] = useState(0);
  const [labels, setLabels] = useState([]);

  const identifier = (images[index] || {}).id;
  const thisImageLabels = labels.filter(l => l.identifier === identifier && !l.is_other_negative).map(l => {
    return {category: l.category, value: l.value};
  });
  const otherNegIsSelected = labels.some(l => l.identifier === identifier && l.is_other_negative);
  const submittable = uniq(labels.map(l => l.identifier)).length === images.length;

  useEffect(() => {
    setIndex(0);
    setLabels([]);
  }, [images]);

  const submit = async () => {
    await submitLabels(labels);
    setIsOpen(false);
  };

  useEffect(() => {
    if (isSubmitting) submit().finally(() => setIsSubmitting(false));
  }, [isSubmitting]);

  //
  // LABELING
  //

  const toggleLabel = (label, type, e) => {
    if (isSubmitting) {
      if (e !== undefined) e.preventDefault();
      return;
    }

    const finalLabel = {...label, identifier, type};

    const newLabels = [...labels];
    const removed = remove(newLabels, l => isEqual(l, finalLabel));
    if (removed.length === 0) {
      // Didn't exist before; we need to add the label
      remove(newLabels, l => l.identifier === identifier && l.type !== type);
      newLabels.push(finalLabel);
    }

    setLabels(newLabels);
    if (e !== undefined) e.preventDefault();
  };

  //
  // KEY BINDINGS
  //

  const handleKeyDown = useCallback((e) => {
    const { key } = e;
    const keyAsNumber = parseInt(key);

    let caught = true;
    if (key === "ArrowLeft") {
      setIndex(Math.max(index - 1, 0));
    } else if (key === "ArrowRight") {
      setIndex(Math.min(index + 1, images.length - 1));
    } else if (key === "ArrowUp") {
      setIsOpen(false);
    } else if (key === "s" && submittable) {
      setIsSubmitting(true);
    } else if (key === "n" && model.augment_negs) {
      toggleLabel({is_other_negative: true}, "other");
    } else if (!isNaN(keyAsNumber) && keyAsNumber >= 1 && keyAsNumber <= model.pos_tags.length) {
      toggleLabel(model.pos_tags[keyAsNumber - 1], "positive");
    } else if (!isNaN(keyAsNumber) &&
               keyAsNumber >= (model.pos_tags.length + 1) &&
               keyAsNumber <= (model.pos_tags.length + model.neg_tags.length)) {
      toggleLabel(model.neg_tags[keyAsNumber - model.pos_tags.length - 1], "negative");
    } else {
      caught = false;
    }

    if (caught) {
      e.preventDefault();
    }
  }, [index, images, submittable, model, toggleLabel]);

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

  const selectedImage = images[index];

  return (
    <Modal
      isOpen={isOpen}
      toggle={() => setIsOpen(false)}
      modalTransition={{ timeout: 25 }}
      backdropTransition={{ timeout: 75 }}
      size="full"
      className={`active-val-modal ${isSubmitting ? "loading" : ""}`}
    >
      {model !== undefined && selectedImage !== undefined && <>
        <ModalHeader toggle={() => setIsOpen(false)}>
          <span>Active validation: image {index + 1} of {images.length}</span>
        </ModalHeader>

        <ModalBody>
          <p>
            <b>Key bindings:</b> &nbsp;
            <kbd>&larr;</kbd> <kbd>&rarr;</kbd> to move between images,{" "}
            <kbd>&uarr;</kbd> to close modal
            {submittable && <>, <kbd>s</kbd> to submit labels</>}
          </p>
          <p>
            <b>Positive labels:</b> &nbsp;
            {model.pos_tags.map((t, i) => {
              const { category, value } = t;
              const isCustom = !BUILT_IN_MODES.some(([v]) => v === value);
              const isSelected = thisImageLabels.some(l => isEqual(l, t));
              return (
                <>
                  {i < 10 && <>
                    <kbd>{(i + 1) % 10}</kbd>{" "}
                  </>}
                  <a
                    href="#"
                    onClick={e => toggleLabel(t, "positive", e)}
                    className={`rbt-token ${isCustom ? "CUSTOM" : value} ${isSelected ? "rbt-token-active" : ""}`}
                  >
                    {category}{isCustom ? ` (${value})` : ""}
                  </a>{" "}
                </>
              );
            })}
          </p>
          <p>
            <b>Negative labels:</b> &nbsp;
            {model.neg_tags.map((t, i) => {
              const { category, value } = t;
              const isCustom = !BUILT_IN_MODES.some(([v]) => v === value);
              const isSelected = thisImageLabels.some(l => isEqual(l, t));
              const j = i + model.pos_tags.length;
              return (
                <>
                  {j < 10 && <>
                    <kbd>{(j + 1) % 10}</kbd>{" "}
                  </>}
                  <a
                    href="#"
                    onClick={e => toggleLabel(t, "negative", e)}
                    className={`rbt-token ${isCustom ? "CUSTOM" : value} ${isSelected ? "rbt-token-active" : ""}`}
                  >
                    {category}{isCustom ? ` (${value})` : ""}
                  </a>{" "}
                </>
              );
            })}
            {model.augment_negs && <>
              <kbd>n</kbd> <a
                href="#"
                onClick={e => toggleLabel({is_other_negative: true}, "other", e)}
                className={`rbt-token CUSTOM ${otherNegIsSelected ? "rbt-token-active" : ""}`}
              >
                (other negative)
              </a>
            </>}
          </p>
          <img className="main w-100" src={selectedImage.src} />
        </ModalBody>
        <ModalFooter>
          <Button
            color="primary"
            disabled={!submittable || isSubmitting}
            onClick={() => setIsSubmitting(true)}
          >Submit</Button>
        </ModalFooter>
      </>}
    </Modal>
  );
}

export default ActiveValidationModal;
