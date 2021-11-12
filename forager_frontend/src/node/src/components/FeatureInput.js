import React, { useEffect, useRef } from "react";
import { Typeahead } from "react-bootstrap-typeahead";

const FeatureInput = ({ features, className, selected, setSelected, noAutofill, ...props }) => {
  const autoFilled = useRef(false);

  useEffect(() => {
    if (!autoFilled.current && !noAutofill && selected.length === 0 && features.length > 0) {
      setSelected([features[0]]);
      autoFilled.current = true;
    }
  }, [noAutofill, features]);

  return (
    <Typeahead
      className={`typeahead-bar ${className || ""}`}
      options={features}
      selected={selected}
      onChange={s => {
        setSelected(s);
      }}
      labelKey="name"
      clearButton
      {...props}
    />
  );
}

export default FeatureInput;
