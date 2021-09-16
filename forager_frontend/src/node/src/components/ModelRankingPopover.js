import React, { useState, useEffect } from "react";
import {
  Popover,
  PopoverBody,
} from "reactstrap";

import FeatureInput from "./FeatureInput";

const ModelRankingPopover = ({ canBeOpen, modelOutputInfo, rankingModel, setRankingModel }) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <Popover
      placement="bottom"
      isOpen={isOpen}
      target="ordering-mode"
      trigger="hover"
      toggle={() => setIsOpen(!isOpen)}
      fade={false}
      popperClassName={`model-ranking-popover ${(canBeOpen && (isOpen || !!!(rankingModel))) ? "visible" : "invisible"}`}
    >
      <PopoverBody>
        <FeatureInput
          id="ranking-feature-bar"
          className="my-1"
          placeholder="Model to rank by"
          features={modelOutputInfo.filter(m => m.has_scores)}
          selected={rankingModel}
          setSelected={setRankingModel}
          noAutofill
        />
      </PopoverBody>
    </Popover>
  );
};

export default ModelRankingPopover;
