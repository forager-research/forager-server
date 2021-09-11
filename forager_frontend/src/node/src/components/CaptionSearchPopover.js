import React, { useState, useEffect } from "react";
import {
  Popover,
  PopoverBody,
  Button,
  Input,
} from "reactstrap";
import Emoji from "react-emoji-render";

import fromPairs from "lodash/fromPairs";
import toPairs from "lodash/toPairs";

const endpoints = fromPairs(toPairs({
  generateTextEmbedding: 'generate_text_embedding',
}).map(([name, endpoint]) => [name, `${process.env.REACT_APP_SERVER_URL}/api/${endpoint}`]));

const CaptionSearchPopover = ({ canBeOpen, text, setText, textEmbedding, setTextEmbedding }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const generateEmbedding = async () => {
    const url = new URL(endpoints.generateTextEmbedding);
    const body = { text };

    const res = await fetch(url, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    }).then(res => res.json());
    setTextEmbedding(res.embedding);
  }

  useEffect(() => {
    if (isLoading) generateEmbedding().finally(() => setIsLoading(false));
  }, [isLoading]);

  return (
    <Popover
      placement="bottom"
      isOpen={true}
      target="ordering-mode"
      trigger="hover"
      toggle={() => setIsOpen(!isOpen)}
      fade={false}
      popperClassName={`caption-search-popover ${isLoading ? "loading" : ""} ${(canBeOpen && (isOpen || isLoading || !!!(textEmbedding))) ? "visible" : "invisible"}`}
    >
      <PopoverBody>
        <div className="mt-1 mb-2">
          Like Google Images... but over your own dataset!
        </div>
        <Input
          autoFocus
          type="textarea"
          value={text}
          onChange={e => {
            setText(e.target.value.replace("\n", " "));
            setTextEmbedding("");
          }}
          placeholder="Caption"
          disabled={isLoading}
        />
        <Button
          color="light"
          onClick={() => setIsLoading(true)}
          disabled={text.trim().length === 0 || isLoading || !!(textEmbedding)}
          className="mt-2 mb-1 w-100"
        >{textEmbedding ? <>
          <Emoji text=":white_check_mark:"/> Ready to query
        </> : "Generate embedding"}</Button>
      </PopoverBody>
    </Popover>
  );
};

export default CaptionSearchPopover;
