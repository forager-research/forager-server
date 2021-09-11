import React, { useState, useEffect } from "react";
import { Typeahead } from "react-bootstrap-typeahead";

import sortBy from "lodash/sortBy";

import NewModeInput from "./NewModeInput";

// TODO(mihirg): Combine with this same constant in other places
const BUILT_IN_MODES = [
  ["POSITIVE", "Positive"],
  ["NEGATIVE", "Negative"],
  ["HARD_NEGATIVE", "Hard Negative"],
  ["UNSURE", "Unsure"],
];

const LabelPanel = ({
  customModesByCategory,
  categoryDispatch,
  category,
  setCategory,
  isVisible,
}) => {
  const setSelected = (selection) => {
    if (selection.length === 0) {
      setCategory(null);
    } else {
      let c = selection[selection.length - 1];
      if (c.customOption) {  // new
        c = c.label;
        categoryDispatch({
          type: "ADD_CATEGORY",
          category: c,
        })
      }
      setCategory(c);
    }
  };

  if (!isVisible) return null;
  return (
    <div className="d-flex flex-row align-items-center justify-content-between">
      <Typeahead
        multiple
        id="label-mode-bar"
        className="typeahead-bar mr-2"
        options={Array.from(customModesByCategory.keys())}
        placeholder="Category to label"
        selected={category ? [category] : []}
        onChange={setSelected}
        newSelectionPrefix="New category: "
        allowNew={true}
      />
      <div className="text-nowrap">
        {BUILT_IN_MODES.map(([value, name], i) =>
          <>
            <kbd>{(i + 1) % 10}</kbd> <span className={`rbt-token ${value}`}>{name.toLowerCase()}</span>&nbsp;
          </>)}
        {!!(category) &&
          (customModesByCategory.get(category) || []).map((name, i) =>
          <>
            <kbd>{(BUILT_IN_MODES.length + i + 1) % 10}</kbd> <span className="rbt-token CUSTOM">{name}</span>&nbsp;
          </>)}
        {!!(category) &&
          <NewModeInput
            category={category}
            customModesByCategory={customModesByCategory}
            categoryDispatch={categoryDispatch}
          />
        }
      </div>
    </div>
  );
};

export default LabelPanel;
