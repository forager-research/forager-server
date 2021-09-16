import React, { useEffect } from "react";
import { Typeahead } from "react-bootstrap-typeahead";

const FeatureInput = ({ features, className, selected, setSelected, noAutofill, ...props }) => {
  useEffect(() => {
    if (!noAutofill && selected === null && features.length > 0) {
      setSelected(features[0]);
    }
  }, [noAutofill, features]);

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
