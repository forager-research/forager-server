import React from "react";
import { Typeahead } from "react-bootstrap-typeahead";

const FeatureInput = ({ features, className, selected, setSelected, ...props }) => {
  return (
    <Typeahead
      className={`typeahead-bar ${className || ""}`}
      options={features}
      selected={selected ? [selected] : []}
      onChange={s => setSelected(s.length === 0 ? null : s[0])}
      labelKey="name"
      clearButton
      {...props}
    />
  );
}

export default FeatureInput;
