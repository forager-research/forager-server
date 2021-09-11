import React, { useState } from "react";
import {
  Input,
} from "reactstrap";

import uniq from "lodash/uniq";

const MAX_CUSTOM_CATEGORIES = 6;

const NewModeInput = ({category, customModesByCategory, categoryDispatch}) => {
  const [value, setValue] = useState("");

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      categoryDispatch({
        type: "ADD_MODE",
        category,
        mode: value,
      });
      setValue("");
    }
    e.stopPropagation();
  };

  return customModesByCategory.get(category).length < MAX_CUSTOM_CATEGORIES ? (
    <Input
      bsSize="sm"
      placeholder="New mode"
      className="new-mode-input"
      value={value}
      onChange={e => setValue(e.target.value)}
      onKeyDown={handleKeyDown}
      onFocus={e => e.stopPropagation()}/>
  ) : <></>;
};

export default NewModeInput;
